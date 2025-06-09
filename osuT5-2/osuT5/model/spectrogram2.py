
import torchaudio
import torch
import torch.nn as nn
import torch.nn.functional as F

class MelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 1024,
        n_mels: int = 384,
        hop_length: int = 128,
    ):
        super().__init__()
        self.transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            n_mels=n_mels,
            hop_length=hop_length,
            center=True,
            f_min=20,
            f_max=sample_rate // 2,
            pad_mode="reflect",
        )

    def forward(self, samples: torch.Tensor) -> torch.Tensor:
        spectrogram = self.transform(samples)
        # Apply logarithmic scaling
        spectrogram = torch.log1p(spectrogram)
        # Permute dimensions if needed
        spectrogram = spectrogram.permute(0, 2, 1)
        return spectrogram