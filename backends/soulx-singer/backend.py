"""LocalAI backend for SoulX-Singer: a score of notes with words in, a sung vocal out.

We answer AudioTransform. `params[score]` is JSON {"notes": [{start, length, pitch, word, slur}], "language", "snap"}
with times in seconds and MIDI pitches. `params[voice]` names a bundled voice (en, zh); with `params[voice]=clip` we
sing in the bundled voice of the language and convert the vocal into the timbre of the uploaded audio. LocalAI always
requires an uploaded audio, so a bundled voice ignores it.
`params[sample_rate]` resamples the output (24000 Hz by default).
"""

import gc
import json
import os
import sys

os.environ["NLTK_DATA"] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "nltk_data")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "common"))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "SoulX-Singer"))

import backend_pb2
import backend_pb2_grpc
import grpc
import librosa
import soundfile
import torch
from grpc_server import run
from singer import SAMPLE_RATE, ScoreNote, Singer, Voice, snap_pitch

code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "SoulX-Singer")

VOICES = {
    "en": Voice(
        os.path.join(code_dir, "example/audio/en_prompt.mp3"), os.path.join(code_dir, "example/audio/en_prompt.json")
    ),
    "zh": Voice(
        os.path.join(code_dir, "example/audio/zh_prompt.mp3"), os.path.join(code_dir, "example/audio/zh_prompt.json")
    ),
}
CLIP = "clip"


def parse_score(text: str) -> tuple[list[ScoreNote], str, bool]:
    score = json.loads(text)
    notes = [
        ScoreNote(
            start=float(note["start"]),
            length=float(note["length"]),
            pitch=int(note["pitch"]),
            word=str(note["word"]),
            slur=bool(note.get("slur", False)),
        )
        for note in score["notes"]
    ]
    return notes, score.get("language", "en"), bool(score.get("snap", True))


class BackendServicer(backend_pb2_grpc.BackendServicer):
    def __init__(self) -> None:
        self.singer: Singer | None = None

    def Health(self, request, context):
        return backend_pb2.Reply(message=b"OK")

    def LoadModel(self, request, context):
        model_dir = os.path.dirname(request.ModelFile or os.path.join(request.ModelPath, request.Model))
        os.environ["SOULX_WHISPER"] = os.path.join(model_dir, "whisper-base")
        try:
            self.singer = Singer(model_dir, code_dir)
        except (FileNotFoundError, KeyError, RuntimeError) as error:
            return backend_pb2.Result(success=False, message=f"could not load SoulX-Singer from {model_dir}: {error}")
        return backend_pb2.Result(success=True, message="SoulX-Singer loaded")

    def AudioTransform(self, request, context):
        if self.singer is None:
            context.abort(grpc.StatusCode.FAILED_PRECONDITION, "SoulX-Singer is not loaded")
        params = dict(request.params)
        if "score" not in params:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "send the notes as params[score]")
        try:
            notes, language, snap = parse_score(params["score"])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"bad score: {error}")
        if not notes:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, "the score has no notes")
        voice = params.get("voice", language)
        if voice != CLIP and voice not in VOICES:
            context.abort(grpc.StatusCode.INVALID_ARGUMENT, f"voice is one of {', '.join([*VOICES, CLIP])}")
        bundled = VOICES.get(language, VOICES["en"]) if voice == CLIP else VOICES[voice]
        sung = self.singer.sing(bundled, notes, language, snap and voice != CLIP)
        if voice == CLIP:
            sung = self.singer.convert(sung, request.audio_path)
            if snap:
                sung = snap_pitch(sung, notes)
        rate = int(params.get("sample_rate", SAMPLE_RATE))
        if rate != SAMPLE_RATE:
            sung = librosa.resample(sung, orig_sr=SAMPLE_RATE, target_sr=rate)
        soundfile.write(request.dst, sung, rate)
        return backend_pb2.AudioTransformResult(dst=request.dst, reference_provided=voice == CLIP)

    def Free(self, request, context):
        self.singer = None
        gc.collect()
        torch.cuda.empty_cache()
        return backend_pb2.Result(success=True)


if __name__ == "__main__":
    run(BackendServicer(), "soulx-singer")
