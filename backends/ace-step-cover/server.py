"""The upstream ace-step backend plus AudioTransform, which covers or continues the uploaded song."""

import argparse

import backend
import backend_pb2
import soundfile
from acestep.constants import TASK_INSTRUCTIONS


def continuation(song_path, seconds):
    """ACE-Step's repaint pads the song when the repainted span ends past it, which continues the song."""
    end_of_song = soundfile.info(song_path).duration
    return {
        "task_type": "repaint",
        "instruction": TASK_INSTRUCTIONS["repaint"],
        "repainting_start": end_of_song,
        "repainting_end": end_of_song + seconds,
    }


def audio_transform(self, request, context):
    params = dict(request.params)
    payload = {
        "task_type": "cover",
        "instruction": TASK_INSTRUCTIONS["cover"],
        "src_audio_path": request.audio_path,
        "prompt": params.get("caption", ""),
        "lyrics": params.get("lyrics", ""),
        "vocal_language": params.get("language", "en"),
        "audio_cover_strength": float(params.get("strength", "0.5")),
        "audio_format": "wav",
        "thinking": False,
    }
    if request.reference_path:
        payload["reference_audio_path"] = request.reference_path
    if params.get("seed"):
        payload["seed"] = int(params["seed"])
    if params.get("task") == "continue":
        payload.update(continuation(request.audio_path, float(params.get("seconds", "60"))))
    backend._generate_audio_sync(self, payload, request.dst)
    return backend_pb2.AudioTransformResult(dst=request.dst, reference_provided=bool(request.reference_path))


backend.BackendServicer.AudioTransform = audio_transform

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--addr", default="localhost:50051", help="Listen address")
    backend.serve(parser.parse_args().addr)
