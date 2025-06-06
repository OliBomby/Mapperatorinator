from __future__ import annotations

import torch
import numpy as np
import numpy.typing as npt

from config import InferenceConfig
from ..dataset.data_utils import load_audio_file, MILISECONDS_PER_SECOND


class Preprocessor(object):
    def __init__(self, args: InferenceConfig, parallel: bool = False):
        """Preprocess audio data into sequences."""
        self.frame_seq_len = args.osut5.data.src_seq_len - 1
        self.frame_size = args.osut5.data.hop_length
        self.sample_rate = args.osut5.data.sample_rate
        self.samples_per_sequence = self.frame_seq_len * self.frame_size
        self.sequence_stride = int(self.samples_per_sequence * (1 - args.lookback - args.lookahead))
        self.parallel = parallel
        if parallel:
            self.sequence_stride = self.samples_per_sequence
        self.miliseconds_per_stride = self.sequence_stride * MILISECONDS_PER_SECOND / self.sample_rate
        self.miliseconds_per_sequence = self.samples_per_sequence * MILISECONDS_PER_SECOND / self.sample_rate
        self.lookback_max_time = args.lookback * self.miliseconds_per_sequence
        self.lookahead_max_time = (1 - args.lookahead) * self.miliseconds_per_sequence
        self.start_time = args.start_time
        self.end_time = args.end_time

    def load(self, path: str) -> npt.ArrayLike:
        """Load an audio file as audio frames. Convert stereo to mono, normalize.

        Args:
            path: Path to audio file.

        Returns:
            samples: Audio time-series.
        """
        return load_audio_file(path, self.sample_rate)

    def segment(
            self,
            samples: npt.ArrayLike,
            begin_pad: int = 0,
            end_pad: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor, float]:
        """Segment audio samples into sequences. Sequences are flattened frames.

        Args:
            samples: Audio time-series.
            begin_pad: Padding at the beginning of the audio in samples.
            end_pad: Padding at the end of the audio in samples.

        Returns:
            sequences: A list of sequences of shape (batch size, samples per sequence).
            sequence_times: A list of sequence start times in miliseconds.
        """
        song_length = len(samples) / self.sample_rate * 1000
        samples = np.pad(samples, [begin_pad, end_pad])

        if len(samples) < self.samples_per_sequence:
            # If samples is smaller than our window size, pad to window size
            padding_needed = self.samples_per_sequence - len(samples)
        else:
            # Calculate padding needed to make the total length exactly fit the striding pattern
            remainder = (len(samples) - self.samples_per_sequence) % self.sequence_stride
            padding_needed = 0 if remainder == 0 else self.sequence_stride - remainder

        samples = np.pad(samples, [0, padding_needed])
        sequences = self.window(samples, self.samples_per_sequence, self.sequence_stride)
        sequences = torch.from_numpy(sequences).to(torch.float32)
        sequence_times = torch.arange(0, len(sequences) * self.miliseconds_per_stride,
                                      self.miliseconds_per_stride).to(torch.int32)

        if self.start_time is not None:
            # Remove all sequences whose generation window is before the start_time
            # If this removes all the sequences, keep the last one
            start_idx = torch.searchsorted(sequence_times, self.start_time - self.lookahead_max_time, right=True)
            if start_idx == len(sequence_times):
                start_idx -= 1
            sequences = sequences[start_idx:]
            sequence_times = sequence_times[start_idx:]
        if self.end_time is not None:
            # Remove all sequences whose generation window is after the end_time
            # If this removes all the sequences, keep the first one
            end_idx = torch.searchsorted(sequence_times, self.end_time - self.lookback_max_time, right=False)
            if end_idx == 0:
                end_idx += 1
            sequences = sequences[:end_idx]
            sequence_times = sequence_times[:end_idx]

        return sequences, sequence_times, song_length

    @staticmethod
    def window(a, w, o, copy=False):
        sh = (a.size - w + 1, w)
        st = a.strides * 2
        view = np.lib.stride_tricks.as_strided(a, strides=st, shape=sh)[0::o]
        if copy:
            return view.copy()
        else:
            return view
