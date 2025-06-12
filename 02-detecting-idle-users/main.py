import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.processors.user_idle_processor import UserIdleProcessor
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.processors.aggregators.openai_llm_context import OpenAILLMContext
from pipecat.services.cartesia.stt import CartesiaSTTService
from pipecat.frames.frames import EndTaskFrame
from pipecat.processors.frame_processor import FrameDirection

load_dotenv(dotenv_path="../.env", override=True)


# ユーザーのアイドル状態を検知した際に実行されるハンドラ
# async def handle_user_idle(processor):
#     # ユーザーがいるか呼びかける
#     await processor.push_frame(TTSSpeakFrame("Are you still there?"))

# アイドル状態の検知回数に応じて処理を分けるパターン
async def handle_user_idle(processor, retry_count: int):
    if retry_count == 1:
        await processor.push_frame(TTSSpeakFrame("Are you still there?"))
        return True # Trueを返した場合はモニタリングを継続
    elif retry_count == 2:
        await processor.push_frame(TTSSpeakFrame("Would you like to continue our conversation?"))
        return True
    else:
        # 3回目のアイドル検知で発話後に会話を終了する
        await processor.push_frame(TTSSpeakFrame("I'll leave you for now. Have a nice day!"))
        # この終了の方法だと実際に上の発話が開始される前に会話のセッションが終了してしまう
        await processor.push_frame(EndTaskFrame(), FrameDirection.UPSTREAM)
        return False # アイドル状態のモニタリングを停止


async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    # Speech to Text にCartesiaSTTを使用する
    stt = CartesiaSTTService(
        api_key=os.getenv("CARTESIA_API_KEY"),
    )

    # ユーザーのアイドル状態を検知するProcessor
    user_idle = UserIdleProcessor(
        # 10秒間ユーザーが発話しない場合にアイドル状態と判断する
        timeout=10,
        # アイドル状態になった際に実行されるハンドラ
        callback=handle_user_idle,
    )

    llm = OpenAILLMService(
        api_key=os.getenv("OPENAI_API_KEY"),
        model="gpt-4o-mini",
    )

    messages = [
        {
            "role": "system",
            "content": "You are Chatbot, a friendly, helpful robot. Your goal is to demonstrate your capabilities in a succinct way. Your output will be converted to audio so don't include special characters in your answers. Respond to what the user said in a creative and helpful way, but keep your responses brief. Start by introducing yourself.",

        }
    ]
    context = OpenAILLMContext(messages=messages)
    context_aggregator = llm.create_context_aggregator(context)


    # Text to Speech にCartesiaを使用する
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="57dcab65-68ac-45a6-8480-6c4c52ec1cd1",
    )

    # テキストが入力されたらTTSを通して音声を出力するパイプラインを作成
    task = PipelineTask(Pipeline([
        transport.input(),
        stt,
        user_idle,
        context_aggregator.user(),
        llm,
        tts,
        transport.output(),
        context_aggregator.assistant(),
    ]))

    # クライアント接続時のハンドラを設定
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # 固定のテキストをTTSを通して音声出力
        await task.queue_frames([TTSSpeakFrame(f"Hello!")])

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    transport_params = {
        "webrtc": lambda: TransportParams(audio_in_enabled=True, audio_out_enabled=True),
    }
    main(run_example, transport_params=transport_params)
