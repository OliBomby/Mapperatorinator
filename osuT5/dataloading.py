import multiprocessing

import hydra
import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import DataLoader

from osuT5.config import TrainConfig
from osuT5.dataset.osu_parser import OsuParser
from osuT5.dataset.ors_dataset import STEPS_PER_MILLISECOND
from osuT5.model.spectrogram import MelSpectrogram
from osuT5.tokenizer import EventType
from osuT5.utils import (
    setup_args,
    get_tokenizer,
    worker_init_fn,
    get_dataset,
)


def play_hs(audio, tokens, sr, tokenizer):
    import sounddevice as sd
    import numpy as np

    # Play audio with hitsounds for every time event in labels
    # Parameters for hitsound
    hitsound_freq = 2000  # Hz
    hitsound_duration = 0.03  # seconds
    hitsound_amp = 0.2

    # Find time events in labels
    time_indices = []
    for t in tokens:
        if tokenizer.event_start[EventType.TIME_SHIFT] <= t < tokenizer.event_end[EventType.TIME_SHIFT]:
            time_event = tokenizer.decode(t.item())
            # Convert to sample index
            x = int(time_event.value / STEPS_PER_MILLISECOND / 1000 * sr)
            time_indices.append(x)

    # Add hitsounds
    audio_with_hits = audio.copy()
    hitsound_samples = int(hitsound_duration * sr)
    t = np.linspace(0, hitsound_duration, hitsound_samples, endpoint=False)
    hitsound = hitsound_amp * np.sin(2 * np.pi * hitsound_freq * t)

    for idx in time_indices:
        end = min(idx + hitsound_samples, len(audio_with_hits))
        audio_with_hits[idx:end] += hitsound[:end - idx]

    sd.play(audio_with_hits, samplerate=sr)


@hydra.main(config_path="../configs/osut5", config_name="train_tiny_dist3", version_base="1.1")
def main(args: TrainConfig):
    setup_args(args)

    mgr = multiprocessing.Manager()
    shared = mgr.Namespace()
    shared.current_train_step = 1
    tokenizer = get_tokenizer(args)
    parser = OsuParser(args, tokenizer)
    dataset = get_dataset(
        args=args,
        test=False,
        parser=parser,
        tokenizer=tokenizer,
        shared=shared,
        # subset_ids=[2933, 1891, 4131],
    )

    dataloader = DataLoader(
        dataset,
        batch_size=args.optim.batch_size,
        num_workers=args.dataloader.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.dataloader.num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    transform = MelSpectrogram(
        args.model.spectrogram.implementation,
        args.model.spectrogram.log_scale,
        args.model.spectrogram.sample_rate,
        args.model.spectrogram.n_fft,
        args.model.spectrogram.n_mels,
        args.model.spectrogram.hop_length,
        f_min=args.model.spectrogram.f_min,
        f_max=args.model.spectrogram.f_max,
    )

    if args.mode == 'benchmark':
        # Iterate one full epoch
        for _ in tqdm.tqdm(dataloader, smoothing=0.01):
            shared.current_train_step += 1
        print(shared.current_train_step)

    if args.mode == 'lengths':
        # Make histogram of the lengths of the sequences
        lengths = []
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            for i in range(len(b["frames"])):  # batch size
                length = b['decoder_attention_mask'][i].sum().item()
                lengths.append(length)
            shared.current_train_step += 1
            if len(lengths) > 100000:
                break

        plt.hist(lengths, bins=100)
        plt.show()

        print(f"Max length: {max(lengths)}")
        print(f"Min length: {min(lengths)}")
        print(f"Mean length: {sum(lengths) / len(lengths)}")
        print(f"Median length: {sorted(lengths)[len(lengths) // 2]}")
        print(f"75th percentile: {sorted(lengths)[len(lengths) * 3 // 4]}")
        print(f"90th percentile: {sorted(lengths)[len(lengths) * 9 // 10]}")
        print(f"95th percentile: {sorted(lengths)[len(lengths) * 19 // 20]}")
        print(f"99th percentile: {sorted(lengths)[len(lengths) * 99 // 100]}")
        print(f"99.9th percentile: {sorted(lengths)[len(lengths) * 999 // 1000]}")
        print(f"99.99th percentile: {sorted(lengths)[len(lengths) * 9999 // 10000]}")
        print(f"99.999th percentile: {sorted(lengths)[len(lengths) * 99999 // 100000]}")

        print(f"Total number of sequences: {len(lengths)}")
        print(f"Total number of tokens: {sum(lengths)}")
        print(f"Total number of sequences with length 0: {lengths.count(2)}")

    if args.mode == 'plot':
        for b in tqdm.tqdm(dataloader, smoothing=0.01):
            mels = transform(b["frames"])
            # [tokenizer.decode(t) if t > 16 else t for t in b['decoder_input_ids'][3].cpu().numpy()]
            # plot the melspectrogram
            play_hs(audio, labels, args, tokenizer)
            for i in range(len(mels)):
                fig, ax = plt.subplots(figsize=(12, 6), dpi=200)
                ax.imshow(mels[i].numpy().T, aspect="auto", origin="lower", norm="log")

                # plot the timing of events as vertical lines
                timings = b["decoder_input_ids"][i]
                start_index = ((timings == tokenizer.sos_id).nonzero(as_tuple=True)[0]).item()
                pre = timings[:start_index]
                post = timings[start_index:]
                for t in pre:
                    if tokenizer.event_start[EventType.TIME_SHIFT] <= t < tokenizer.event_end[EventType.TIME_SHIFT]:
                        time_event = tokenizer.decode(t.item())
                        x = time_event.value / STEPS_PER_MILLISECOND / 1000 * args.model.spectrogram.sample_rate / args.model.spectrogram.hop_length
                        ax.vlines(x=x, ymin=0, ymax=mels[i].shape[1], color='g')
                for t in post:
                    if tokenizer.event_start[EventType.TIME_SHIFT] <= t < tokenizer.event_end[EventType.TIME_SHIFT]:
                        time_event = tokenizer.decode(t.item())
                        x = time_event.value / STEPS_PER_MILLISECOND / 1000 * args.model.spectrogram.sample_rate / args.model.spectrogram.hop_length
                        ax.vlines(x=x, ymin=0, ymax=mels[i].shape[1] / 20, color='b')
                labels = b["labels"][i]
                for t in labels:
                    if tokenizer.event_start[EventType.TIME_SHIFT] <= t < tokenizer.event_end[EventType.TIME_SHIFT]:
                        time_event = tokenizer.decode(t.item())
                        x = time_event.value / STEPS_PER_MILLISECOND / 1000 * args.model.spectrogram.sample_rate / args.model.spectrogram.hop_length
                        ax.vlines(x=x, ymin=0, ymax=mels[i].shape[1] / 10, color='r')

                plt.show()
            break

if __name__ == "__main__":
    main()
