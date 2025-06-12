import argparse
import os

from dotenv import load_dotenv
from loguru import logger

from pipecat.frames.frames import EndFrame, TTSSpeakFrame
from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.transports.base_transport import BaseTransport, TransportParams

load_dotenv(dotenv_path="../.env", override=True)




async def run_example(transport: BaseTransport, _: argparse.Namespace, handle_sigint: bool):
    logger.info(f"Starting bot")

    # Text to Speech にCartesiaを使用する
    tts = CartesiaTTSService(
        api_key=os.getenv("CARTESIA_API_KEY"),
        voice_id="0b32066b-2bcc-44b9-89ab-0223a09d1606",
    )

    # テキストが入力されたらTTSを通して音声を出力するパイプラインを作成
    task = PipelineTask(Pipeline([tts, transport.output()]))

    # クライアント接続時のハンドラを設定
    @transport.event_handler("on_client_connected")
    async def on_client_connected(transport, client):
        # 固定のテキストをTTSを通して音声を出力し、終了する
        await task.queue_frames([TTSSpeakFrame(f"Hello!"), EndFrame()])

    runner = PipelineRunner(handle_sigint=handle_sigint)

    await runner.run(task)


if __name__ == "__main__":
    from pipecat.examples.run import main

    transport_params = {
        "webrtc": lambda: TransportParams(audio_out_enabled=True),
    }
    main(run_example, transport_params=transport_params)
