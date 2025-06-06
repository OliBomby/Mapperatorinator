import time
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from torch import nn
from tqdm import tqdm

import routed_pickle
from osuT5.config import TrainConfig
from osuT5.dataset.ors_dataset import STEPS_PER_MILLISECOND, LABEL_IGNORE_ID
from osuT5.model import Mapperatorinator
from osuT5.tokenizer import ContextType
from osuT5.utils import calc_loss, get_stats
from osuT5.tokenizer import EventType, Tokenizer
from osuT5.utils import (
    setup_args,
    get_model,
    get_dataloaders, Averager, add_prefix, acc_range, fuzzy_acc_range,
    get_shared_training_state,
)

logger = get_logger(__name__)


def load_model(
        ckpt_path_str: str,
        t5_args: TrainConfig,
        device,
):
    if ckpt_path_str == "":
        raise ValueError("Model path is empty.")

    ckpt_path = Path(ckpt_path_str)

    def tokenizer_loader():
        if not (ckpt_path / "pytorch_model.bin").exists() or not (ckpt_path / "custom_checkpoint_0.pkl").exists():
            tokenizer = Tokenizer.from_pretrained(ckpt_path_str)
        else:
            tokenizer_state = torch.load(ckpt_path / "custom_checkpoint_0.pkl", pickle_module=routed_pickle, weights_only=False)
            tokenizer = Tokenizer()
            tokenizer.load_state_dict(tokenizer_state)
        return tokenizer

    tokenizer = tokenizer_loader()

    def model_loader():
        if not (ckpt_path / "pytorch_model.bin").exists() or not (ckpt_path / "custom_checkpoint_0.pkl").exists():
            model = Mapperatorinator.from_pretrained(ckpt_path_str)
            model.generation_config.disable_compile = True
        else:
            model_state = torch.load(ckpt_path / "pytorch_model.bin", map_location=device, weights_only=True)
            model = get_model(t5_args, tokenizer)
            model.load_state_dict(model_state)

        model.eval()
        model.to(device)
        return model

    return model_loader(), tokenizer


def test(args: TrainConfig, accelerator: Accelerator, model, tokenizer, preprefix: str):
    setup_args(args)

    shared = get_shared_training_state()
    shared.current_train_step = args.optim.total_steps

    _, test_dataloader = get_dataloaders(tokenizer, args, shared)

    # noinspection PyTypeChecker
    test_dataloader = accelerator.prepare(test_dataloader)

    class_weights = torch.ones(tokenizer.vocab_size_out)
    class_weights[tokenizer.event_start[EventType.TIME_SHIFT]:tokenizer.event_end[EventType.TIME_SHIFT]] = args.data.rhythm_weight
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=LABEL_IGNORE_ID)
    loss_fn = loss_fn.to(accelerator.device)

    all_in_contexts = set()
    for cts in args.data.context_types:
        if isinstance(cts, str):
            all_in_contexts.add(cts)
        else:
            all_in_contexts.update(cts["in"])

    with torch.no_grad():
        start_time = time.time()
        averager = Averager()

        max_time = 1000 * args.data.src_seq_len * args.model.spectrogram.hop_length / args.model.spectrogram.sample_rate
        n_bins = 100
        bins = np.linspace(0, max_time, n_bins + 1)[1:]
        bin_totals = {}
        bin_counts = {}
        max_rhythm_complexity = 4
        rhythm_complexity_n_bins = 20
        rhythm_complexity_bins = np.linspace(0, max_rhythm_complexity, rhythm_complexity_n_bins + 1)[1:]
        rhythm_complexity_bin_totals = {}
        rhythm_complexity_bin_counts = {}
        fuzzy_rhythm_complexity_bin_totals = {}
        precision_bin_range = 3
        precision_bins = np.arange(-precision_bin_range, precision_bin_range + 1)
        precision_bin_totals = {}
        precision_bin_counts = {}

        for batch_id, batch in enumerate(tqdm(test_dataloader), start=1):  # type: int, dict[str, torch.Tensor]
            if batch_id == args.eval.steps * args.optim.grad_acc:
                break

            rhythm_complexity: Optional[torch.Tensor] = None
            if "sample_weights" in batch:
                rhythm_complexity = batch["sample_weights"]

            # We can't use the beatmap idx of the test set because these are not known by the model
            del batch["beatmap_idx"]

            outputs = model(**batch)
            loss = outputs.loss
            preds = torch.argmax(outputs.logits, dim=-1)
            labels = batch["labels"]

            def gather_metrics(loss, preds, labels, rhythm_complexity=None, prefix=''):
                # Calculate accuracy metrics
                stats = get_stats(loss, preds, labels, tokenizer, args)

                if prefix != '':
                    stats = add_prefix(prefix, stats)

                averager.update(stats)

                # Initialize bin totals for this prefix
                if prefix not in bin_totals:
                    bin_totals[prefix] = np.zeros(n_bins)
                    bin_counts[prefix] = np.zeros(n_bins)
                    rhythm_complexity_bin_totals[prefix] = np.zeros(rhythm_complexity_n_bins)
                    rhythm_complexity_bin_counts[prefix] = np.zeros(rhythm_complexity_n_bins)
                    fuzzy_rhythm_complexity_bin_totals[prefix] = np.zeros(rhythm_complexity_n_bins)
                    precision_bin_totals[prefix] = np.zeros(2 * precision_bin_range + 1)
                    precision_bin_counts[prefix] = np.zeros(2 * precision_bin_range + 1)

                # Calculate timing precision histogram
                index = (tokenizer.event_start[EventType.TIME_SHIFT] <= labels) & (labels < tokenizer.event_end[EventType.TIME_SHIFT])
                range_labels = labels[index]
                range_preds = preds[index]
                timing_diffs = (range_preds - range_labels).detach().cpu().numpy()
                for i, n in enumerate(precision_bins):  # type: int, int
                    accs = timing_diffs == n
                    precision_bin_totals[prefix][i] += np.sum(accs)
                    # noinspection PyTypeChecker
                    precision_bin_counts[prefix][i] += len(accs)

                # Bin labels by time and calculate accuracy
                preds = preds.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
                label_times = np.empty(labels.shape, dtype=np.float32)
                for j in range(len(labels)):
                    t = 0
                    for i, l in enumerate(labels[j]):
                        if tokenizer.event_start[EventType.TIME_SHIFT] <= l < tokenizer.event_end[EventType.TIME_SHIFT]:
                            time_event = tokenizer.decode(l)
                            t = time_event.value / STEPS_PER_MILLISECOND
                        label_times[j, i] = t

                binned_labels = np.digitize(label_times, bins)
                for i in range(n_bins):
                    bin_preds = preds[binned_labels == i]
                    bin_labels = labels[binned_labels == i]
                    index = (bin_labels != LABEL_IGNORE_ID) & (bin_labels != tokenizer.eos_id)
                    bin_preds = bin_preds[index]
                    bin_labels = bin_labels[index]
                    bin_totals[prefix][i] += np.sum(bin_preds == bin_labels)
                    bin_counts[prefix][i] += len(bin_preds)

                # Bin timing accuracy by rhythm complexity
                if rhythm_complexity is not None:
                    rhythm_complexity = rhythm_complexity.cpu().numpy()
                    binned_rhythm_complexity = np.digitize(rhythm_complexity, rhythm_complexity_bins)
                    for i in range(len(rhythm_complexity)):
                        sample_bin = np.clip(binned_rhythm_complexity[i], 0, n_bins - 1)
                        sample = acc_range(preds[i], labels[i], tokenizer.event_start[EventType.TIME_SHIFT],
                                           tokenizer.event_end[EventType.TIME_SHIFT])
                        fuzzy_sample = fuzzy_acc_range(preds[i], labels[i], tokenizer.event_start[EventType.TIME_SHIFT],
                                                       tokenizer.event_end[EventType.TIME_SHIFT], 2)
                        rhythm_complexity_bin_totals[prefix][sample_bin] += np.sum(sample)
                        rhythm_complexity_bin_counts[prefix][sample_bin] += len(sample)
                        fuzzy_rhythm_complexity_bin_totals[prefix][sample_bin] += np.sum(fuzzy_sample)

            if len(args.data.context_types) > 0:
                for cts in args.data.context_types:
                    if isinstance(cts, str):
                        cts = {"out": [ContextType.MAP], "in": [cts]}

                    ct_index = torch.ones_like(batch['decoder_input_ids'][:, 0], dtype=torch.bool)
                    for c in cts["in"]:
                        ct_index &= torch.max(batch['decoder_input_ids'] ==
                                              tokenizer.context_sos[c], dim=1).values
                    for c in all_in_contexts - set(cts["in"]):
                        ct_index &= ~torch.max(batch['decoder_input_ids'] ==
                                               tokenizer.context_sos[c], dim=1).values

                    if not ct_index.any():
                        continue

                    ct_logits = outputs.logits[ct_index]
                    ct_preds = preds[ct_index]
                    ct_labels = labels[ct_index]
                    ct_weights = batch["sample_weights"][ct_index]
                    ct_rhythm_complexity = rhythm_complexity[ct_index] if rhythm_complexity is not None else None
                    ct_loss = calc_loss(loss_fn, ct_logits, ct_labels, ct_weights)

                    # noinspection PyUnresolvedReferences
                    c_prefix = f"{'+'.join(c.value for c in cts['in'])}"
                    gather_metrics(ct_loss, ct_preds, ct_labels, ct_rhythm_complexity, prefix=c_prefix)
            else:
                gather_metrics(loss, preds, labels, rhythm_complexity)

        def plot_bins(bin_totals, bin_counts, bins, y_name, x_name, prefixes):
            for prefix in reversed(prefixes):
                if prefix == '':
                    continue

                y_name = f"{prefix}/{y_name}"

            bin_accs = bin_totals / bin_counts
            wandb.define_metric(y_name, step_metric=x_name)

            # Log the plot
            for (bin_x, bin_acc) in zip(bins, bin_accs):
                if not np.isnan(bin_acc):
                    wandb.log({y_name: bin_acc, x_name: bin_x})

        for prefix in bin_totals.keys():
            prefixes = [preprefix, prefix]

            # Plot timing over rhythm complexity
            if rhythm_complexity_bin_counts[prefix].sum() > 0:
                plot_bins(rhythm_complexity_bin_totals[prefix], rhythm_complexity_bin_counts[prefix], rhythm_complexity_bins,
                          "timing_acc_over_rhythm_complexity", "rhythm_complexity", prefixes)
                plot_bins(fuzzy_rhythm_complexity_bin_totals[prefix], rhythm_complexity_bin_counts[prefix], rhythm_complexity_bins,
                          "fuzzy_timing_acc_over_rhythm_complexity", "rhythm_complexity", prefixes)

            # Plot timing precision
            plot_bins(precision_bin_totals[prefix], precision_bin_counts[prefix], precision_bins, "timing_precision", "offset", prefixes)

            # Plot bin accuracies
            plot_bins(bin_totals[prefix], bin_counts[prefix], bins, "acc_over_time", "bin_time", prefixes)

        averager.update({"time": time.time() - start_time})
        averaged_stats = averager.average()
        averaged_stats = add_prefix(preprefix, averaged_stats)
        accelerator.log(averaged_stats)
        logger.info(averaged_stats)


@hydra.main(config_path="../configs/osut5", config_name="train_tiny_dist", version_base="1.1")
def main(args: TrainConfig):
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="", logging_dir="tensorboard_logs"
        ),
    )
    accelerator.init_trackers(
        "osuT5",
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                "job_type": "testing",
            }
        }
    )

    model, tokenizer = load_model(args.checkpoint_path, args, "cuda")
    model.eval()

    # noinspection PyTypeChecker
    model = accelerator.prepare(model)

    args.data.sample_weights_path = "../../../datasets/rhythm_complexities.csv"
    args.data.timing_random_offset = 0
    args.data.dt_augment_prob = 0
    test(args, accelerator, model, tokenizer, "test")

    args.data.timing_random_offset = 2
    test(args, accelerator, model, tokenizer, "test_noise")


if __name__ == "__main__":
    main()
