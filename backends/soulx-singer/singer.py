"""Sings a score (notes with words) with SoulX-Singer in a bundled voice, snaps each note's pitch onto the score, and
can then convert the vocal into the timbre of any voice clip with SoulX-Singer's SVC model."""

import copy
import itertools
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
LENGTH_TOLERANCE = 0.03
LONG_NOTE_SECONDS = 0.7
IN_TUNE_SEMITONES = 0.35
SPLICE_SECONDS = 0.025
FADE_IN_SECONDS = 0.04
FADE_OUT_SECONDS = 0.04
PEAK_LEVEL = 0.89
PITCH_GLIDE_FRAMES = 9
TAKES = 3
CLICK_RATIO = 6
CLICK_FLOOR = 0.01
CRACK_DB = 9
GAP_DB = 25
GAP_SECONDS = 0.1
ONSET_DB = 30
MAX_ENTRY_SHIFT_SECONDS = 0.2
LEGAL_ONSETS = {
    "B L",
    "B R",
    "D R",
    "D W",
    "F L",
    "F R",
    "G L",
    "G R",
    "K L",
    "K R",
    "K W",
    "P L",
    "P R",
    "S K",
    "S L",
    "S M",
    "S N",
    "S P",
    "S T",
    "S W",
    "SH R",
    "T R",
    "T W",
    "TH R",
    "S K R",
    "S P R",
    "S T R",
    "S P L",
    "S K W",
}


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


def syllables(phones: list[str]) -> list[list[str]]:
    """Splits a word's phones into syllables, giving each vowel the longest legal onset before it."""
    vowels = [index for index, phone in enumerate(phones) if phone[-1].isdigit()]
    cuts = []
    for left, right in itertools.pairwise(vowels):
        between = phones[left + 1 : right]
        onset = next(
            (size for size in range(min(3, len(between)), 1, -1) if " ".join(between[-size:]) in LEGAL_ONSETS),
            min(1, len(between)),
        )
        cuts.append(right - onset)
    bounds = [0, *cuts, len(phones)]
    return [phones[start:end] for start, end in itertools.pairwise(bounds)]


def syllables_on(phones: list[str], notes: int) -> list[list[str]]:
    """A word's syllables for the notes it is sung on; the last note sings every syllable the notes run out for."""
    parts = syllables(phones)
    if len(parts) <= notes:
        return parts
    return [*parts[: notes - 1], [phone for part in parts[notes - 1 :] for phone in part]]


def split_words(segment: dict) -> dict:
    """Gives each note of a word sung over several notes its own syllable. SoulX-Singer reads a word on several notes
    as one word sung on the first note and held over the rest, so "every" on three notes came out as "every-ee-ee"."""
    words, phones, kinds = segment["text"].split(), segment["phoneme"].split(), segment["note_type"].split()
    first = 0
    while first < len(words):
        last = first + 1
        while last < len(words) and kinds[last] == "3" and words[last] == words[first]:
            last += 1
        if last - first > 1 and phones[first].startswith("en_"):
            for index, part in enumerate(syllables_on(phones[first][3:].split("-"), last - first)):
                phones[first + index] = "en_" + "-".join(part)
                kinds[first + index] = "2"
        first = last
    return {**segment, "phoneme": " ".join(phones), "note_type": " ".join(kinds)}


def glitches(audio: np.ndarray, notes: list[ScoreNote]) -> int:
    """Counts clicks, sample jumps far above their neighbourhood, and cracks, sudden level drops inside a held note."""
    step = np.abs(np.diff(audio))
    window = int(0.01 * SAMPLE_RATE)
    spikes = (step > CLICK_RATIO * np.convolve(step, np.ones(window) / window, mode="same") + CLICK_FLOOR).astype(int)
    hop = int(FRAME_PERIOD_MS / 1000 * SAMPLE_RATE)
    level = 20 * np.log10(np.sqrt(np.convolve(audio**2, np.ones(hop) / hop, mode="same")[::hop]) + 1e-5)
    cracks = 0
    for note in notes:
        held = level[
            int((note.start + 0.06) * 1000 / FRAME_PERIOD_MS) : int(
                (note.start + note.length - 0.06) * 1000 / FRAME_PERIOD_MS
            )
        ]
        cracks += int((np.diff(held) < -CRACK_DB).sum())
    return int((np.diff(spikes) == 1).sum()) + 2 * cracks + 3 * gaps(level, notes)


def gaps(level: np.ndarray, notes: list[ScoreNote]) -> int:
    """Counts notes the singer drops out on for a moment, as when "the radio" came out as "the ... radio"."""
    voiced = level[level > level.max() - 40]
    floor = (np.median(voiced) if len(voiced) else level.max()) - GAP_DB
    run = round(GAP_SECONDS * 1000 / FRAME_PERIOD_MS)
    count = 0
    for note in notes:
        first = int((note.start + 0.05) * 1000 / FRAME_PERIOD_MS)
        last = int((note.start + note.length - 0.05) * 1000 / FRAME_PERIOD_MS)
        inside = level[first:last]
        if len(inside) >= run:
            quiet = np.convolve((inside < floor).astype(int), np.ones(run, dtype=int), mode="valid")
            count += int(quiet.max() >= run)
    return count


def voice_onset(audio: np.ndarray) -> float:
    """Seconds from the start of a rendered line to the singer's first sound."""
    hop = int(0.002 * SAMPLE_RATE)
    head = audio[:SAMPLE_RATE]
    level = 20 * np.log10(np.sqrt(np.convolve(head**2, np.ones(hop) / hop, mode="same")[::hop]) + 1e-9)
    return float(np.argmax(level > level.max() - ONSET_DB)) * hop / SAMPLE_RATE


def keep_long_notes(fitted: np.ndarray, raw: np.ndarray, notes: list[ScoreNote]) -> np.ndarray:
    """Puts SoulX-Singer's own audio back on long held notes it already sang in tune. WORLD's resynthesis keeps short
    notes on pitch but crackles on a sustained vowel, so a held note it did not need to fix sounds better untouched."""
    signal = raw.astype(np.float64)
    f0, times = pyworld.dio(signal, SAMPLE_RATE, frame_period=FRAME_PERIOD_MS)
    mixed = fitted[: len(raw)].copy()
    edge = int(SAMPLE_RATE * SPLICE_SECONDS)
    for note in notes:
        if note.length < LONG_NOTE_SECONDS:
            continue
        inside = (times >= note.start) & (times < note.start + note.length) & (f0 > 0)
        if inside.sum() < MIN_VOICED_FRAMES:
            continue
        if abs(12 * np.log2(np.median(f0[inside]) / midi_to_hz(note.pitch))) > IN_TUNE_SEMITONES:
            continue
        begin, end = int(note.start * SAMPLE_RATE), min(len(raw), int((note.start + note.length) * SAMPLE_RATE))
        if end - begin <= 2 * edge:
            continue
        weight = np.ones(end - begin, dtype=np.float32)
        weight[:edge] = np.linspace(0, 1, edge, dtype=np.float32)
        weight[-edge:] = np.linspace(1, 0, edge, dtype=np.float32)
        mixed[begin:end] = mixed[begin:end] * (1 - weight) + raw[begin:end] * weight
    return mixed


def as_long_as(audio: np.ndarray, seconds: float) -> np.ndarray:
    length = int(seconds * SAMPLE_RATE)
    return np.pad(audio, (0, max(0, length - len(audio))))[:length].astype(np.float32)


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
    position = np.linspace(0, len(f0) - 1, frames)
    low = np.floor(position).astype(int)
    high = np.minimum(low + 1, len(f0) - 1)
    weight = (position - low)[:, None]
    envelope = envelope[low] * (1 - weight) + envelope[high] * weight
    aperiodicity = aperiodicity[low] * (1 - weight) + aperiodicity[high] * weight
    voiced = (f0[low] > 0) & (f0[high] > 0)
    blended = np.exp(
        np.log(np.maximum(f0[low], 1)) * (1 - weight[:, 0]) + np.log(np.maximum(f0[high], 1)) * weight[:, 0]
    )
    f0 = np.where(voiced, blended, f0[low])
    if snap:
        f0 = snap_pitch(f0, notes)
    sung = pyworld.synthesize(
        np.ascontiguousarray(f0),
        np.ascontiguousarray(envelope),
        np.ascontiguousarray(aperiodicity),
        SAMPLE_RATE,
        FRAME_PERIOD_MS,
    )
    return sung[: int(seconds * SAMPLE_RATE)].astype(np.float32)


def snap_pitch(f0: np.ndarray, notes: list[ScoreNote]) -> np.ndarray:
    """Moves each note's median pitch onto the score, gliding between corrections over a few frames, because a
    correction that jumps at a note start clicks."""
    times = np.arange(len(f0)) * FRAME_PERIOD_MS / 1000
    shift = np.full(len(f0), np.nan)
    for note in notes:
        span = (times >= note.start) & (times < note.start + note.length)
        inside = span & (f0 > 0)
        if inside.sum() < MIN_VOICED_FRAMES:
            continue
        miss = 12 * np.log2(np.median(f0[inside]) / midi_to_hz(note.pitch))
        if abs(miss) <= SNAP_LIMIT_SEMITONES:
            shift[span] = -miss
    known = ~np.isnan(shift)
    if not known.any():
        return f0
    shift = np.interp(np.arange(len(f0)), np.flatnonzero(known), shift[known])
    pad = PITCH_GLIDE_FRAMES // 2
    shift = np.convolve(np.pad(shift, pad, mode="edge"), np.ones(PITCH_GLIDE_FRAMES) / PITCH_GLIDE_FRAMES, mode="valid")
    return np.where(f0 > 0, f0 * 2 ** (shift / 12), 0)


def fade_edges(audio: np.ndarray) -> np.ndarray:
    """Fades a rendered line in and out over a few milliseconds, so lines placed side by side never click."""
    fade_in, fade_out = int(SAMPLE_RATE * FADE_IN_SECONDS), int(SAMPLE_RATE * FADE_OUT_SECONDS)
    faded = audio.copy()
    if len(faded) > fade_in + fade_out:
        faded[:fade_in] *= np.linspace(0, 1, fade_in, dtype=np.float32) ** 2
        faded[-fade_out:] *= np.linspace(1, 0, fade_out, dtype=np.float32)
    return faded


def normalize_peak(audio: np.ndarray) -> np.ndarray:
    """Scales a whole part so its loudest moment sits just under full scale; held vowels otherwise clip the WAV."""
    peak = float(np.max(np.abs(audio))) if len(audio) else 0.0
    return audio * (PEAK_LEVEL / peak) if peak > 0 else audio


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
                inside = [
                    ScoreNote(note.start - offset - begin, note.length, note.pitch, note.word, note.slur)
                    for note in line
                    if begin <= note.start - offset < finish
                ]
                takes = [self.take(prompt, split_words(segment), seed, inside, snap) for seed in range(TAKES)]
                fitted = min(takes, key=lambda audio: glitches(audio, inside))
                # SoulX-Singer starts a line 50 to 120 ms after its first note, so we move the line up to sing on
                # the beat, and the fade-in then softens the singer's first sound rather than the silence before it.
                late = min(MAX_ENTRY_SHIFT_SECONDS, voice_onset(fitted))
                fitted = fade_edges(fitted[int(late * SAMPLE_RATE) :])
                start = int((offset + begin) * SAMPLE_RATE)
                length = min(len(fitted), len(audio) - start)
                audio[start : start + length] = fitted[:length]
        return normalize_peak(audio)

    def take(self, prompt: dict, segment: dict, seed: int, notes: list[ScoreNote], snap: bool) -> np.ndarray:
        """Sings one segment with a fixed seed and fits it to the score. SoulX-Singer sings differently on every
        seed, so we sing a few takes and keep the cleanest."""
        seconds = (segment["time"][1] - segment["time"][0]) / 1000
        torch.manual_seed(seed)
        with torch.no_grad():
            sung = self.model.infer(
                {"prompt": prompt, "target": self.processor.process(copy.deepcopy(segment), None)},
                auto_shift=False,
                pitch_shift=0,
                n_steps=self.config.infer.n_steps,
                cfg=self.config.infer.cfg,
                control="score",
                use_fp16=False,
            )
        rendered = sung.squeeze().float().cpu().numpy()
        fitted = fit_to_score(rendered, seconds, notes, snap)
        if abs(len(rendered) / SAMPLE_RATE - seconds) <= seconds * LENGTH_TOLERANCE:
            fitted = keep_long_notes(fitted, as_long_as(rendered, seconds), notes)
        return as_long_as(fitted, seconds)

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
