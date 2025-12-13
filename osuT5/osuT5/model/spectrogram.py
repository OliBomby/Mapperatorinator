from __future__ import annotations

import torch
import torch.nn as nn


class MelSpectrogram(nn.Module):
    def __init__(
        self,
        implementation: str = "nnAudio",
        log_scale: bool = False,
        sample_rate: int = 16000,
        n_ftt: int = 2048,
        n_mels: int = 512,
        hop_length: int = 128,
        f_min: int = 0,
        f_max: int = 8000,
        pad_mode: str = "constant",
    ):
        """
        Melspectrogram transformation layer, supports on-the-fly processing on GPU.

        Attributes:
            implementation: The implementation to use, either "torchaudio" or "nnAudio".
            log_scale: Whether to apply log scaling to the output.
            sample_rate: The sampling rate for the input audio.
            n_ftt: The window size for the STFT.
            n_mels: The number of Mel filter banks.
            hop_length: The hop (or stride) size.
            f_min: The minimum frequency for the Mel filter banks.
            f_max: The maximum frequency for the Mel filter banks.
            pad_mode: The padding mode for the STFT.
        """
        super().__init__()
        assert implementation in ["torchaudio", "nnAudio"], f"Unsupported implementation: {implementation}"
        self.log_scale = log_scale

        if implementation == "torchaudio":
            import torchaudio.transforms
            self.transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_ftt,
                n_mels=n_mels,
                hop_length=hop_length,
                center=True,
                f_min=f_min,
                f_max=f_max,
                pad_mode=pad_mode,
            )
        elif implementation == "nnAudio":
            from nnAudio import features
            self.transform = features.MelSpectrogram(
                sr=sample_rate,
                n_fft=n_ftt,
                n_mels=n_mels,
                hop_length=hop_length,
                center=True,
                fmin=f_min,
                fmax=f_max,
                pad_mode=pad_mode,
            )

    def forward(self, samples: torch.tensor) -> torch.tensor:
        """
        Convert a batch of audio frames into a batch of Mel spectrogram frames.

        For each item in the batch:
        1. pad left and right ends of audio by n_fft // 2.
        2. run STFT with window size of |n_ftt| and stride of |hop_length|.
        3. convert result into mel-scale.
        4. therefore, n_frames = n_samples // hop_length + 1.

        Args:
            samples: Audio time-series (batch size, n_samples).

        Returns:
            A batch of Mel spectrograms of size (batch size, n_frames, n_mels).
        """
        spectrogram = self.transform(samples)
        if self.log_scale:
            spectrogram = torch.log1p(spectrogram)
        spectrogram = spectrogram.permute(0, 2, 1)
        return spectrogram
