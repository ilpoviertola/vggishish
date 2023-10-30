"""
I am so sorry for this code. I wrote it in a hurry and it's a mess. I'll clean it up later...
(maybe :D). This code will extract greatest hit actions from the audio files and save them
so VGGIshIsh classifier can be tasked with classifying them.

This is also pretty unefficient since no multiprocessing stuff is implemented...

Raises:
    NotImplementedError: For now only supports extracting greatest hit actions...
"""
from dataclasses import dataclass
from math import floor, ceil
from pathlib import Path
from argparse import ArgumentParser, Namespace
import json

import torchaudio
import librosa
import numpy as np
from torch import Tensor
from torchvision.transforms import Compose
from tqdm import tqdm


SR = 24000  # default desired sampling rate
GREATEST_HIT_ACTION_TYPES = ["scratch", "hit"]


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--sr", type=int, default=SR)
    parser.add_argument("--output", type=str, default=".")
    parser.add_argument("--filename_mask", type=str, default="*.wav")
    parser.add_argument(
        "--extract_gh_actions",
        default=False,
        action="store_true",
    )
    parser.add_argument("--gh_meta_filename_mask", type=str, default="*_times.txt")
    parser.add_argument("--gh_action_buffer", type=float, default=0.5)
    return parser.parse_args()


class MelSpectrogram:
    def __init__(self, sr, nfft, fmin, fmax, nmels, hoplen, spec_power, inverse=False):
        self.sr = sr
        self.nfft = nfft
        self.fmin = fmin
        self.fmax = fmax
        self.nmels = nmels
        self.hoplen = hoplen
        self.spec_power = spec_power
        self.inverse = inverse

        self.mel_basis = librosa.filters.mel(
            sr=sr, n_fft=nfft, fmin=fmin, fmax=fmax, n_mels=nmels
        )

    def __call__(self, x):
        if self.inverse:
            spec = librosa.feature.inverse.mel_to_stft(
                x,
                sr=self.sr,
                n_fft=self.nfft,
                fmin=self.fmin,
                fmax=self.fmax,
                power=self.spec_power,
            )
            wav = librosa.griffinlim(spec, hop_length=self.hoplen)
            return wav
        else:
            spec = (
                np.abs(librosa.stft(x, n_fft=self.nfft, hop_length=self.hoplen))
                ** self.spec_power
            )
            mel_spec = np.dot(self.mel_basis, spec)
            return mel_spec


class LowerThresh:
    def __init__(self, min_val, inverse=False):
        self.min_val = min_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.maximum(self.min_val, x)


class Add:
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x - self.val
        else:
            return x + self.val


class Subtract:
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x + self.val
        else:
            return x - self.val


class Multiply:
    def __init__(self, val, inverse=False) -> None:
        self.val = val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x / self.val
        else:
            return x * self.val


class Divide:
    def __init__(self, val, inverse=False):
        self.inverse = inverse
        self.val = val

    def __call__(self, x):
        if self.inverse:
            return x * self.val
        else:
            return x / self.val


class Log10:
    def __init__(self, inverse=False):
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return 10**x
        else:
            return np.log10(x)


class Clip:
    def __init__(self, min_val, max_val, inverse=False):
        self.min_val = min_val
        self.max_val = max_val
        self.inverse = inverse

    def __call__(self, x):
        if self.inverse:
            return x
        else:
            return np.clip(x, self.min_val, self.max_val)


def get_mel_spec_transform(sample_rate: int = SR):
    return Compose(
        [
            MelSpectrogram(
                sr=sample_rate,
                nfft=1024,
                fmin=125,
                fmax=7600,
                nmels=80,
                hoplen=1024 // 4,
                spec_power=1,
            ),
            LowerThresh(1e-5),
            Log10(),
            Multiply(20),
            Subtract(20),
            Add(100),
            Divide(100),
            Clip(0, 1.0),
        ]
    )


@dataclass
class GreatestHitActionAudio:
    file_path: Path
    occuring_time: float
    action_type: GREATEST_HIT_ACTION_TYPES
    sample_rate: int = SR
    buffer: float = 0.5

    def __post_init__(self):
        assert (
            self.file_path.exists()
        ), f"File {self.file_path.as_posix()} doesn't exist"
        assert (
            self.file_path.suffix == ".wav" or self.file_path.suffix == ".mp3"
        ), f"File {self.file_path.as_posix()} is not an audio file"

        self.audio_info = torchaudio.info(self.file_path)

        assert self.occuring_time <= self.full_audio_duration, (
            f"occuring_time {self.occuring_time} > full_audio_duration "
            f"{self.full_audio_duration}"
        )
        assert self.occuring_time >= 0, f"occuring_time {self.occuring_time} < 0"

        assert (
            self.action_type.lower() in GREATEST_HIT_ACTION_TYPES
        ), f"Invalid action type {self.action_type.lower()} for {self.file_path.as_posix()}"

        assert (
            self.sample_rate > 0
        ), f"Invalid sample rate {self.sample_rate}, must be > 0"

        assert self.buffer >= 0, f"Invalid buffer {self.buffer}, must be >= 0"

        if self.orig_sample_rate != self.sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                self.orig_sample_rate, self.sample_rate
            )

        self.melspec_transform = get_mel_spec_transform(self.sample_rate)
        self.audio_cached = None
        self.melspec_cached = None

    @property
    def full_audio_duration(self) -> float:
        return self.audio_info.num_frames / self.audio_info.sample_rate

    @property
    def orig_sample_rate(self) -> int:
        return self.audio_info.sample_rate

    @property
    def occuring_time_as_samples(self) -> int:
        return floor(self.occuring_time * self.sample_rate)

    @property
    def buffer_as_samples(self) -> int:
        return ceil(self.buffer * self.sample_rate)

    @property
    def action_start_as_samples(self) -> int:
        if floor(self.occuring_time_as_samples - self.buffer_as_samples) < 0:
            return 0
        return floor(self.occuring_time_as_samples - self.buffer_as_samples)

    @property
    def action_end_as_samples(self) -> int:
        if (
            ceil(self.occuring_time_as_samples + self.buffer_as_samples)
            > self.audio_info.num_frames
        ):
            return self.audio_info.num_frames
        return ceil(self.occuring_time_as_samples + self.buffer_as_samples)

    @property
    def action(self):
        return self.action_type.lower()

    @property
    def audio(self) -> Tensor:
        if self.audio_cached is None:
            audio, _ = torchaudio.load(self.file_path)
            if self.orig_sample_rate != self.sample_rate:
                audio = self.resampler(audio)
            self.audio_cached = audio[
                ..., self.action_start_as_samples : self.action_end_as_samples
            ]
        return self.audio_cached

    @property
    def melspec(self) -> np.ndarray:
        if self.melspec_cached is None:
            self.melspec_cached = self.melspec_transform(self.audio.squeeze(0).cpu().numpy())
        return self.melspec_cached

    def save_audio(self, outputdir: Path, force: bool = False):
        basename = self.file_path.stem.split("_")[0]
        savepath = (
            outputdir / "audio" / f"{basename}_{self.occuring_time_as_samples}.wav"
        )

        if savepath.exists() and not force:
            print(f"File {savepath.as_posix()} already exists, skipping")
            return

        savepath.parent.mkdir(parents=True, exist_ok=True)
        torchaudio.save(savepath, self.audio, self.sample_rate)

    def save_melspec(self, outputdir: Path, force: bool = False):
        basename = self.file_path.stem.split("_")[0]
        savepath = (
            outputdir / "melspecs" / f"{basename}_{self.occuring_time_as_samples}.npy"
        )

        if savepath.exists() and not force:
            print(f"File {savepath.as_posix()} already exists, skipping")
            return

        savepath.parent.mkdir(parents=True, exist_ok=True)
        np.save(savepath, self.melspec)


def extract_gh_actions(
    meta_file: Path, output_dir: Path, time_pad_around_action: float = 0.5
):
    with open(meta_file, encoding="utf-8") as f:
        lines = f.read().splitlines()

    basename = meta_file.stem.split("_")[0]
    gha_labels = {}

    for line in lines:
        occuring_time, _, action_type, _ = line.split(
            " "
        )  # occuring_time, material, action_type, effect
        occuring_time = float(occuring_time)
        action_type = action_type.lower()
        try:
            gha = GreatestHitActionAudio(
                file_path=meta_file.parent / f"{basename}_denoised.wav",
                occuring_time=occuring_time,
                action_type=action_type,
                buffer=time_pad_around_action,
            )
            gha.save_audio(output_dir)
            gha.save_melspec(output_dir)
            gha_labels[f"{basename}_{gha.occuring_time_as_samples}"] = gha.action
        except AssertionError as e:
            # print(e)
            continue

    return gha_labels


def run_pipeline(args: Namespace):
    input_path = Path(args.input)
    output_path = Path(args.output)
    assert input_path.exists(), f"Input path {input_path.as_posix()} doesn't exist"

    if args.extract_gh_actions:
        print("extracting greatest hit actions")
        meta_files = list(input_path.glob(args.gh_meta_filename_mask))
        action_audios = {}
        for f in tqdm(meta_files):
            labels = extract_gh_actions(f, output_path, args.gh_action_buffer)
            action_audios.update(labels)
        with open(output_path / "labels.json", "w", encoding="utf-8") as f:
            json.dump(action_audios, f, indent=4)
    else:
        if input_path.is_dir():
            files = list(input_path.glob(args.filename_mask))
            assert (
                len(files) > 0
            ), f"No files found in {input_path.as_posix()} with mask {args.filename_mask}"
        else:
            files = [input_path]
        raise NotImplementedError("Only GreatestHit per action extraction support :)")

    # init resampler
    # orig_sr = torchaudio.info(files[0]).sample_rate  # assume every file has same sr
    # resampler = torchaudio.transforms.Resample(orig_sr, args.sr)
    # print(f"resampling from {orig_sr}Hz to {args.sr}Hz")

    # wav, orig_sr = torchaudio.load(args.input)


if __name__ == "__main__":
    args = get_args()
    run_pipeline(args)
