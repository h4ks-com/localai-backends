"""LocalAI backend for LTX-2.5 distilled image to video, with an optional two-stage x2 latent upscale.

The model config sets `model` to a transformer GGUF and the `diffusers_dir` option to a local copy of
Lightricks/LTX-2.5-Diffusers. Requests set `params[stages]` to 2 (default: half size, x2 latent upscale, then a
short pass at full size; width and height must be multiples of 64) or 1 (full size directly). The distilled model
runs a fixed sigma schedule without guidance, so step and cfg_scale have no effect.
"""

import ctypes
import gc
import os
import sys

import backend_pb2
import backend_pb2_grpc
import grpc
import torch
import transformers
from diffusers import (
    GGUFQuantizationConfig,
    LTX2ImageToVideoPipeline,
    LTX2LatentUpsamplePipeline,
    LTX2VideoTransformer3DModel,
)
from diffusers.loaders.single_file_model import SINGLE_FILE_LOADABLE_CLASSES
from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
from diffusers.utils import encode_video, load_image
from PIL import ImageOps

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "common"))
from grpc_server import run
from python_utils import parse_options

DEFAULT_FPS = 24
LIBC = ctypes.CDLL("libc.so.6")
LTX2_SINGLE_FILE = SINGLE_FILE_LOADABLE_CLASSES["LTX2VideoTransformer3DModel"]
CONVERT_LTX2 = LTX2_SINGLE_FILE["checkpoint_mapping_fn"]
# GGUF files keep LTX-2.5's original names for the prompt modulation MLPs, so we rename them to the names
# diffusers loads, or those layers stay empty.
PROMPT_ADALN_RENAMES = (
    ("audio_prompt_adaln_single.", "audio_prompt_adaln."),
    ("prompt_adaln_single.", "prompt_adaln."),
)


def convert_ltx25(checkpoint, **kwargs):
    converted = CONVERT_LTX2(checkpoint, **kwargs)
    for key in list(converted):
        for old, new in PROMPT_ADALN_RENAMES:
            if key.startswith(old):
                converted[new + key[len(old) :]] = converted.pop(key)
                break
    return converted


LTX2_SINGLE_FILE["checkpoint_mapping_fn"] = convert_ltx25


def release_cached_memory() -> None:
    """We hand back the pinned buffers PyTorch keeps from offloading, so the process stays flat across clips."""
    gc.collect()
    torch.cuda.empty_cache()
    torch._C._host_emptyCache()
    # Clips run on a gRPC worker thread whose glibc heap keeps freed pages until we trim it.
    LIBC.malloc_trim(0)


def valid_frame_count(frames: int) -> int:
    """LTX needs 8k+1 frames."""
    return max(9, (frames - 1) // 8 * 8 + 1)


class Pipelines:
    def __init__(self, transformer_path: str, diffusers_dir: str) -> None:
        transformer = LTX2VideoTransformer3DModel.from_single_file(
            transformer_path,
            quantization_config=GGUFQuantizationConfig(compute_dtype=torch.bfloat16),
            config=os.path.join(diffusers_dir, "transformer"),
            torch_dtype=torch.bfloat16,
        )
        # The 12B text encoder takes 24 GB in bf16; in 4 bits it fits on a 24 GB card beside the transformer.
        text_encoder = transformers.Gemma4UnifiedForConditionalGeneration.from_pretrained(
            os.path.join(diffusers_dir, "text_encoder"),
            quantization_config=transformers.BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
            ),
            torch_dtype=torch.bfloat16,
        )
        self.video = LTX2ImageToVideoPipeline.from_pretrained(
            diffusers_dir,
            transformer=transformer,
            text_encoder=text_encoder,
            prompt_enhancer=None,
            duration_head=None,
            torch_dtype=torch.bfloat16,
        )
        # The whole pipeline needs more than 24 GB, so we move each part onto the card only while it runs.
        self.video.enable_model_cpu_offload()
        self.video.vae.enable_tiling()
        upsampler = LTX2LatentUpsamplerModel.from_pretrained(
            diffusers_dir, subfolder="latent_upsampler", torch_dtype=torch.bfloat16
        )
        self.upsample = LTX2LatentUpsamplePipeline(vae=self.video.vae, latent_upsampler=upsampler.to("cuda"))

    def generate(self, request, fps: int) -> tuple:
        params = dict(request.params)
        frames = valid_frame_count(request.num_frames or 121)
        width, height = request.width or 576, request.height or 1024
        shared = {
            # The pipeline stretches the first frame to the video size, so we crop it to that shape first.
            "image": ImageOps.fit(load_image(request.start_image), (width, height)),
            "prompt": request.prompt,
            "frame_rate": float(fps),
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            "modality_scale": 1.0,
            "audio_modality_scale": 1.0,
            "generator": torch.Generator("cuda").manual_seed(request.seed or 42),
            "return_dict": False,
        }
        two_stages = params.get("stages", "2") == "2" and width % 64 == 0 and height % 64 == 0
        if not two_stages:
            return self.video(
                width=width,
                height=height,
                num_frames=frames,
                sigmas=DISTILLED_SIGMA_VALUES,
                output_type="np",
                **shared,
            )
        latents, audio_latents = self.video(
            width=width // 2,
            height=height // 2,
            num_frames=frames,
            sigmas=DISTILLED_SIGMA_VALUES,
            output_type="latent",
            **shared,
        )
        upsampled = self.upsample(latents=latents, output_type="latent", return_dict=False)[0]
        return self.video(
            num_frames=frames,
            sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
            latents=upsampled,
            audio_latents=audio_latents,
            noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
            output_type="np",
            **shared,
        )


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self) -> None:
        self.pipelines: Pipelines | None = None

    def Health(self, request, context):
        return backend_pb2.Reply(message=b"OK")

    def LoadModel(self, request, context):
        transformer_path = request.ModelFile or os.path.join(request.ModelPath, request.Model)
        diffusers_dir = os.path.join(request.ModelPath, str(parse_options(request.Options).get("diffusers_dir", "")))
        try:
            self.pipelines = Pipelines(transformer_path, diffusers_dir)
        except (OSError, ValueError, RuntimeError) as error:
            return backend_pb2.Result(success=False, message=f"could not load LTX-2.5: {error}")
        return backend_pb2.Result(success=True, message="LTX-2.5 loaded")

    def GenerateVideo(self, request, context):
        if self.pipelines is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "LTX-2.5 is not loaded")
        if not request.start_image:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "send the first frame as start_image")
        fps = request.fps or DEFAULT_FPS
        video, audio = self.pipelines.generate(request, fps)
        encode_video(
            video[0],
            fps=fps,
            output_path=request.dst,
            audio=audio[0].float().cpu(),
            audio_sample_rate=self.pipelines.video.vocoder.config.output_sampling_rate,
        )
        del video, audio
        release_cached_memory()
        return backend_pb2.Result(success=True, message=request.dst)

    def Free(self, request, context):
        self.pipelines = None
        release_cached_memory()
        return backend_pb2.Result(success=True)


if __name__ == "__main__":
    run(BackendServicer(), "ltx2")
