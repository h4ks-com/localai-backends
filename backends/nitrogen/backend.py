"""LocalAI backend for NVIDIA NitroGen: one game frame in, the next chunk of gamepad states out.

LocalAI has no call for game actions, so we answer Predict: the frame arrives as the chat
message's image and the reply text is the chunk as JSON.
"""

import base64
import gc
import io
import json
import os
import sys
import time

import backend_pb2
import backend_pb2_grpc
import grpc
import numpy as np
import torch
from PIL import Image
from transformers import AutoImageProcessor

# NitroGen parses DEBUG as an integer when imported, and LocalAI starts backends with DEBUG=true.
os.environ["DEBUG"] = "0"
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from grpc_server import run
from nitrogen.cfg import CkptConfig
from nitrogen.flow_matching_transformer.nitrogen import NitroGen
from nitrogen.inference_session import InferenceSession
from nitrogen.mm_tokenizers import NitrogenTokenizer
from nitrogen.shared import BUTTON_ACTION_TOKENS

FRAME_SIZE = (256, 256)
STEPS_PER_SECOND = 60
SIGLIP_DIR = "siglip-large-patch16-256"


def siglip_dir(checkpoint_path: str) -> str:
    """Fine-tuned checkpoints live one folder below the base one and share its SigLIP."""
    folder = os.path.dirname(checkpoint_path)
    for candidate in (folder, os.path.dirname(folder)):
        if os.path.isdir(os.path.join(candidate, SIGLIP_DIR)):
            return os.path.join(candidate, SIGLIP_DIR)
    raise FileNotFoundError(f"no {SIGLIP_DIR} next to {checkpoint_path}")


def load_session(checkpoint_path: str) -> InferenceSession:
    """Load NitroGen offline: SigLIP from the folder next to the checkpoint, no game-ID table."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = CkptConfig.model_validate(checkpoint["ckpt_config"])
    config.model_cfg.vision_encoder_name = siglip_dir(checkpoint_path)
    config.tokenizer_cfg.training = False
    # The game-ID table is built from NVIDIA's training parquet files, which are not public.
    config.tokenizer_cfg.game_mapping_cfg = None
    tokenizer = NitrogenTokenizer(config.tokenizer_cfg)
    model = NitroGen(config=config.model_cfg, game_mapping=None)
    weights = {k: v for k, v in checkpoint["model"].items() if not k.startswith("game_embedding")}
    model.load_state_dict(weights)
    model.eval()
    tokenizer.eval()
    model.to("cuda")
    return InferenceSession(
        model,
        checkpoint_path,
        tokenizer,
        AutoImageProcessor.from_pretrained(config.model_cfg.vision_encoder_name),
        config,
        {},
        None,
        False,
        1.0,
        1,
        1,
    )


def frame_bytes(image: str) -> io.BytesIO:
    """LocalAI hands images over as a file path, a data URI or bare base64."""
    if os.path.isfile(image):
        with open(image, "rb") as file:
            return io.BytesIO(file.read())
    return io.BytesIO(base64.b64decode(image.split(",", 1)[-1]))


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self) -> None:
        self.session: InferenceSession | None = None

    def Health(self, request, context):
        return backend_pb2.Reply(message=b"OK")

    def LoadModel(self, request, context):
        path = request.ModelFile or os.path.join(request.ModelPath, request.Model)
        try:
            self.session = load_session(path)
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            return backend_pb2.Result(success=False, message=f"could not load NitroGen from {path}: {error}")
        return backend_pb2.Result(success=True, message="NitroGen loaded")

    def Predict(self, request, context):
        if self.session is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "NitroGen is not loaded")
        if not request.Images:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "send the game frame as the message image")
        image = Image.open(frame_bytes(request.Images[-1])).convert("RGB")
        start = time.monotonic()
        result = self.session.predict(np.asarray(image.resize(FRAME_SIZE)))
        chunk = {
            "fps": STEPS_PER_SECOND,
            "names": BUTTON_ACTION_TOKENS,
            "buttons": np.asarray(result["buttons"]).round(3).tolist(),
            "j_left": np.asarray(result["j_left"]).round(3).tolist(),
            "j_right": np.asarray(result["j_right"]).round(3).tolist(),
            "seconds": round(time.monotonic() - start, 3),
        }
        return backend_pb2.Reply(message=json.dumps(chunk).encode())

    def Free(self, request, context):
        self.session = None
        gc.collect()
        torch.cuda.empty_cache()
        return backend_pb2.Result(success=True)


if __name__ == "__main__":
    run(BackendServicer(), "nitrogen")
