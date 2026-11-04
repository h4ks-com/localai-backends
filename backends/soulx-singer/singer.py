"""Sings a score (notes with words) with SoulX-Singer in a bundled voice, snaps each note's pitch onto the score, and
can then convert the vocal into the timbre of any voice clip with SoulX-Singer's SVC model."""

import json
import os
import tempfile
from dataclasses import dataclass

import librosa
import numpy as np
import pyworld
import soundfile
import torch
from preprocess.tools.f0_extraction import F0Extractor
from preprocess.tools.midi_parser import Note, notes2meta
from soulxsinger.models.soulxsinger import SoulXSinger
from soulxsinger.models.soulxsinger_svc import SoulXSingerSVC
from soulxsinger.utils.data_processor import DataProcessor
from soulxsinger.utils.file_utils import load_config

SAMPLE_RATE = 24000
FRAME_PERIOD_MS = 5.0
LANGUAGES = {"en": "English", "zh": "Mandarin", "yue": "Cantonese"}
MAX_VOICE_SECONDS = 20
MIN_VOICED_FRAMES = 2
PHRASE_GAP_SECONDS = 0.15
SNAP_LIMIT_SEMITONES = 2.0


@dataclass
class ScoreNote:
    start: float
    length: float
    pitch: int
    word: str
    slur: bool = False


@dataclass
class Voice:
    wav: str
    meta: str


@dataclass
class Converter:
    model: SoulXSingerSVC
    f0: F0Extractor


def midi_to_hz(pitch: float) -> float:
    return 440.0 * 2 ** ((pitch - 69) / 12)


def phrases(notes: list[ScoreNote]) -> list[list[ScoreNote]]:
    """Splits the score at every breath, so each sung line is rendered and timed on its own."""
    groups: list[list[ScoreNote]] = []
    for note in notes:
        if groups and note.start - (groups[-1][-1].start + groups[-1][-1].length) < PHRASE_GAP_SECONDS:
            groups[-1].append(note)
        else:
            groups.append([note])
    return groups


def fit_to_score(audio: np.ndarray, seconds: float, notes: list[ScoreNote], snap: bool) -> np.ndarray:
    """Stretches a rendered stretch of singing to the length the score gives it, so no line drifts off the beat, and
    with snap moves each note's median pitch onto the score when the singer missed by a little. A bigger miss is a
    slide or a different note, and forcing it that far strains the voice, so we leave it."""
    signal = audio.astype(np.float64)
    f0, times = pyworld.dio(signal, SAMPLE_RATE, frame_period=FRAME_PERIOD_MS)
    f0 = pyworld.stonemask(signal, f0, times, SAMPLE_RATE)
    envelope = pyworld.cheaptrick(signal, f0, times, SAMPLE_RATE)
    aperiodicity = pyworld.d4c(signal, f0, times, SAMPLE_RATE)
    frames = max(2, round(seconds * 1000 / FRAME_PERIOD_MS) + 1)
    source = np.round(np.linspace(0, len(f0) - 1, frames)).astype(int)
    f0, envelope, aperiodicity = f0[source].copy(), envelope[source].copy(), aperiodicity[source].copy()
    if snap:
        times = np.arange(frames) * FRAME_PERIOD_MS / 1000
        for note in notes:
            inside = (times >= note.start) & (times < note.start + note.length) & (f0 > 0)
            if inside.sum() < MIN_VOICED_FRAMES:
                continue
            miss = 12 * np.log2(np.median(f0[inside]) / midi_to_hz(note.pitch))
            if abs(miss) <= SNAP_LIMIT_SEMITONES:
                f0[inside] = f0[inside] * 2 ** (-miss / 12)
    sung = pyworld.synthesize(
        np.ascontiguousarray(f0),
        np.ascontiguousarray(envelope),
        np.ascontiguousarray(aperiodicity),
        SAMPLE_RATE,
        FRAME_PERIOD_MS,
    )
    return sung[: int(seconds * SAMPLE_RATE)].astype(np.float32)


def load_weights(model: torch.nn.Module, path: str, device: str) -> torch.nn.Module:
    checkpoint = torch.load(path, weights_only=False, map_location="cpu")
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


class Singer:
    def __init__(self, model_dir: str, code_dir: str, device: str = "cuda") -> None:
        self.config = load_config(os.path.join(code_dir, "soulxsinger/config/soulxsinger.yaml"))
        self.device = device
        self.model = load_weights(SoulXSinger(self.config), os.path.join(model_dir, "model.pt"), device)
        self.processor = DataProcessor(
            hop_size=self.config.audio.hop_size,
            sample_rate=self.config.audio.sample_rate,
            phoneset_path=os.path.join(code_dir, "soulxsinger/utils/phoneme/phone_set.json"),
            device=device,
        )
        self.model_dir = model_dir
        self.converter: Converter | None = None

    def sing(self, voice: Voice, notes: list[ScoreNote], language: str, snap: bool) -> np.ndarray:
        with open(voice.meta, encoding="utf-8") as handle:
            prompt = self.processor.process(json.load(handle)[0], voice.wav)
        end = max(note.start + note.length for note in notes)
        audio = np.zeros(int(end * SAMPLE_RATE) + SAMPLE_RATE // 2, dtype=np.float32)
        for line in phrases(notes):
            offset = line[0].start
            for segment in self.segments(line, offset, language):
                begin, finish = segment["time"][0] / 1000, segment["time"][1] / 1000
                with torch.no_grad():
                    sung = self.model.infer(
                        {"prompt": prompt, "target": self.processor.process(segment, None)},
                        auto_shift=False,
                        pitch_shift=0,
                        n_steps=self.config.infer.n_steps,
                        cfg=self.config.infer.cfg,
                        control="score",
                        use_fp16=False,
                    )
                inside = [
                    ScoreNote(note.start - offset - begin, note.length, note.pitch, note.word, note.slur)
                    for note in line
                    if begin <= note.start - offset < finish
                ]
                fitted = fit_to_score(sung.squeeze().float().cpu().numpy(), finish - begin, inside, snap)
                start = int((offset + begin) * SAMPLE_RATE)
                length = min(len(fitted), len(audio) - start)
                audio[start : start + length] = fitted[:length]
        return audio

    @staticmethod
    def segments(line: list[ScoreNote], offset: float, language: str) -> list[dict]:
        """SoulX-Singer's metadata for one line, with times from the line's first note."""
        with tempfile.TemporaryDirectory() as workdir:
            target_path = os.path.join(workdir, "target.json")
            score = [
                Note(
                    start_s=note.start - offset,
                    note_dur=note.length,
                    note_text=note.word,
                    note_pitch=note.pitch,
                    note_type=3 if note.slur else 2,
                )
                for note in line
            ]
            notes2meta(score, target_path, None, LANGUAGES.get(language, "English"), None)
            with open(target_path, encoding="utf-8") as handle:
                return json.load(handle)

    def convert(self, audio: np.ndarray, voice_clip: str) -> np.ndarray:
        """Re-voices a sung vocal in the timbre of the clip, keeping its pitch and timing."""
        if self.converter is None:
            self.converter = Converter(
                load_weights(SoulXSingerSVC(self.config), os.path.join(self.model_dir, "model-svc.pt"), self.device),
                F0Extractor(
                    os.path.join(self.model_dir, "rmvpe.pt"),
                    device=self.device,
                    hop_size=self.config.audio.hop_size,
                    verbose=False,
                ),
            )
        converter = self.converter
        clip, _ = librosa.load(voice_clip, sr=SAMPLE_RATE, duration=MAX_VOICE_SECONDS)
        with tempfile.TemporaryDirectory() as workdir:
            clip_path = os.path.join(workdir, "clip.wav")
            sung_path = os.path.join(workdir, "sung.wav")
            soundfile.write(clip_path, clip, SAMPLE_RATE)
            soundfile.write(sung_path, audio, SAMPLE_RATE)
            clip_f0 = converter.f0.process(clip_path)
            sung_f0 = converter.f0.process(sung_path)
        with torch.no_grad():
            converted, _ = converter.model.infer(
                pt_wav=torch.from_numpy(clip).unsqueeze(0).to(self.device),
                gt_wav=torch.from_numpy(audio).unsqueeze(0).to(self.device),
                pt_f0=torch.from_numpy(clip_f0).unsqueeze(0).to(self.device),
                gt_f0=torch.from_numpy(sung_f0).unsqueeze(0).to(self.device),
                auto_shift=False,
                pitch_shift=0,
                n_steps=self.config.infer.n_steps,
                cfg=self.config.infer.cfg,
                use_fp16=False,
            )
        return converted.squeeze().float().cpu().numpy()[: len(audio)]
