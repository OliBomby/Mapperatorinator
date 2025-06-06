import json
from pathlib import Path
import numpy.typing as npt

import hydra
import torch
from omegaconf import DictConfig
from slider import Beatmap
from torch.utils.data import IterableDataset

from classifier.libs.dataset import OsuParser
from classifier.libs.dataset.data_utils import load_audio_file
from classifier.libs.dataset.ors_dataset import STEPS_PER_MILLISECOND
from classifier.libs.model.model import OsuClassifierOutput
from classifier.libs.tokenizer import Tokenizer, Event, EventType
from classifier.libs.utils import load_ckpt


def iterate_examples(
        beatmap: Beatmap,
        audio: npt.NDArray,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device
):
    frame_seq_len = model_args.data.src_seq_len - 1
    frame_size = model_args.data.hop_length
    sample_rate = model_args.data.sample_rate
    samples_per_sequence = frame_seq_len * frame_size

    parser = OsuParser(model_args, tokenizer)
    events, event_times = parser.parse(beatmap)

    for sample in range(0, len(audio) - samples_per_sequence, samples_per_sequence):
        example = create_example(events, event_times, audio, sample / sample_rate, model_args, tokenizer, device)
        yield example


class ExampleDataset(IterableDataset):
    def __init__(self, beatmap, audio, classifier_args, classifier_tokenizer, device):
        self.beatmap = beatmap
        self.audio = audio
        self.classifier_args = classifier_args
        self.classifier_tokenizer = classifier_tokenizer
        self.device = device

    def __iter__(self):
        return iterate_examples(
            self.beatmap,
            self.audio,
            self.classifier_args,
            self.classifier_tokenizer,
            self.device
        )


def create_example(
        events: list[Event],
        event_times: list[float],
        audio: npt.NDArray,
        time: float,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device,
        unsqueeze: bool = False,
):
    frame_seq_len = model_args.data.src_seq_len - 1
    frame_size = model_args.data.hop_length
    sample_rate = model_args.data.sample_rate
    samples_per_sequence = frame_seq_len * frame_size
    sequence_duration = samples_per_sequence / sample_rate

    # Get audio frames
    frame_start = int(time * sample_rate)
    frames = audio[frame_start:frame_start + samples_per_sequence]
    frames = torch.from_numpy(frames).to(torch.float32).to(device)

    # Get the events between time and time + sequence_duration
    events = [event for event, event_time in zip(events, event_times) if
              time <= event_time / 1000 < time + sequence_duration]
    # Normalize time shifts
    for i, event in enumerate(events):
        if event.type == EventType.TIME_SHIFT:
            events[i] = Event(EventType.TIME_SHIFT, int((event.value - time * 1000) * STEPS_PER_MILLISECOND))

    # Tokenize the events
    tokens = torch.full((model_args.data.tgt_seq_len,), tokenizer.pad_id, dtype=torch.long)
    for i in range(min(len(events), model_args.data.tgt_seq_len)):
        tokens[i] = tokenizer.encode(events[i])
    tokens = tokens.to(device)

    if unsqueeze:
        tokens = tokens.unsqueeze(0)
        frames = frames.unsqueeze(0)

    return {
        "decoder_input_ids": tokens,
        "decoder_attention_mask": tokens != tokenizer.pad_id,
        "frames": frames,
    }


def create_example_from_path(
        beatmap_path: str,
        audio_path: str,
        time: float,
        model_args: DictConfig,
        tokenizer: Tokenizer,
        device: torch.device,
        unsqueeze: bool = False,
):
    sample_rate = model_args.data.sample_rate

    beatmap_path = Path(beatmap_path)
    beatmap = Beatmap.from_path(beatmap_path)

    # Get audio frames
    if audio_path == '':
        audio_path = beatmap_path.parent / beatmap.audio_filename

    audio = load_audio_file(audio_path, sample_rate)

    parser = OsuParser(model_args, tokenizer)
    events, event_times = parser.parse(beatmap)

    return create_example(events, event_times, audio, time, model_args, tokenizer, device, unsqueeze)


def get_mapper_names(path: str):
    path = Path(path)

    # Load JSON data from file
    with open(path, 'r') as file:
        data = json.load(file)

    # Populate beatmap_mapper
    mapper_names = {}
    for item in data:
        if len(item['username']) == 0:
            mapper_name = "Unknown"
        else:
            mapper_name = item['username'][0]
        mapper_names[item['user_id']] = mapper_name

    return mapper_names


@hydra.main(config_path="configs", config_name="inference", version_base="1.1")
def main(args: DictConfig):
    torch.set_grad_enabled(False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, model_args, tokenizer = load_ckpt(args.checkpoint_path)
    model.eval().to(device)

    example = create_example_from_path(args.beatmap_path, args.audio_path, args.time, model_args, tokenizer, device, True)
    result: OsuClassifierOutput = model(**example)
    logits = result.logits

    # Print the top 100 mappers with confidences
    top_k = 100
    top_k_indices = logits[0].topk(top_k).indices
    top_k_confidences = logits[0].topk(top_k).values

    mapper_idx_id = {idx: ids for ids, idx in tokenizer.mapper_idx.items()}
    mapper_names = get_mapper_names(args.mappers_path)

    for idx, confidence in zip(top_k_indices, top_k_confidences):
        mapper_id = mapper_idx_id[idx.item()]
        mapper_name = mapper_names.get(mapper_id, "Unknown")
        print(f"Mapper: {mapper_name} ({mapper_id}) with confidence: {confidence.item()}")


if __name__ == "__main__":
    main()
