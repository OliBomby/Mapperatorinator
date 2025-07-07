import logging
import os
import random
import traceback
from datetime import timedelta
from pathlib import Path
from typing import Optional

import hydra
import numpy as np
import torch
from scipy import linalg
from slider import Beatmap, Circle, Slider, Spinner, HoldNote
from torch.utils.data import DataLoader
from tqdm import tqdm

from classifier.classify import ExampleDataset
from classifier.libs.model.model import OsuClassifierOutput
from classifier.libs.utils import load_ckpt
from config import FidConfig
from inference import prepare_args, load_diff_model, generate, load_model
from osuT5.osuT5.dataset.data_utils import load_audio_file, load_mmrs_metadata, filter_mmrs_metadata
from osuT5.osuT5.inference import generation_config_from_beatmap, beatmap_config_from_beatmap
from osuT5.osuT5.tokenizer import ContextType
from multiprocessing import Manager, Process

logger = logging.getLogger(__name__)


def get_beatmap_paths(args: FidConfig) -> list[Path]:
    """Get all beatmap paths (.osu) from the dataset directory."""
    dataset_path = Path(args.dataset_path)

    if args.dataset_type == "mmrs":
        metadata = load_mmrs_metadata(dataset_path)
        filtered_metadata = filter_mmrs_metadata(
            metadata,
            start=args.dataset_start,
            end=args.dataset_end,
            gamemodes=args.gamemodes,
        )
        beatmap_files = [dataset_path / "data" / item["BeatmapSetFolder"] / item["BeatmapFile"] for _, item in filtered_metadata.iterrows()]
    elif args.dataset_type == "ors":
        beatmap_files = []
        track_names = ["Track" + str(i).zfill(5) for i in range(args.dataset_start, args.dataset_end)]
        for track_name in track_names:
            for beatmap_file in (dataset_path / track_name / "beatmaps").iterdir():
                beatmap_files.append(dataset_path / track_name / "beatmaps" / beatmap_file.name)
    else:
        raise ValueError(f"Unknown dataset type: {args.dataset_type}")

    return beatmap_files


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Numpy implementation of the Frechet Distance.
    The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
    and X_2 ~ N(mu_2, C_2) is
            d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).

    Stable version by Dougal J. Sutherland.

    Params:
    -- mu1   : Numpy array containing the activations of a layer of the
               inception net (like returned by the function 'get_predictions')
               for generated samples.
    -- mu2   : The sample mean over activations, precalculated on an
               representative data set.
    -- sigma1: The covariance matrix over activations for generated samples.
    -- sigma2: The covariance matrix over activations, precalculated on an
               representative data set.

    Returns:
    --   : The Frechet Distance.
    """

    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert (
        mu1.shape == mu2.shape
    ), "Training and test mean vectors have different lengths"
    assert (
        sigma1.shape == sigma2.shape
    ), "Training and test covariances have different dimensions"

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = (
            "fid calculation produces singular product; "
            "adding %s to diagonal of cov estimates"
        ) % eps
        logger.warning(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError("Imaginary component {}".format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def add_to_dict(source_dict, target_dict):
    for key, value in source_dict.items():
        if key not in target_dict:
            target_dict[key] = value
        else:
            target_dict[key] += value


def calculate_rhythm_stats(real_rhythm, generated_rhythm):
    # Rhythm is a set of timestamps for each beat
    # Calculate number of true positives, false positives, and false negatives within a leniency of 10 ms
    leniency = 10
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    for real_beat in real_rhythm:
        if any(abs(real_beat - gen_beat) <= leniency for gen_beat in generated_rhythm):
            true_positives += 1
        else:
            false_negatives += 1

    for gen_beat in generated_rhythm:
        if not any(abs(gen_beat - real_beat) <= leniency for real_beat in real_rhythm):
            false_positives += 1

    return {
        "true_positives": true_positives,
        "false_positives": false_positives,
        "false_negatives": false_negatives,
    }


def calculate_precision(rhythm_stats):
    true_positives = rhythm_stats["true_positives"]
    false_positives = rhythm_stats["false_positives"]
    if true_positives + false_positives == 0:
        return 0.0
    return true_positives / (true_positives + false_positives)


def calculate_recall(rhythm_stats):
    true_positives = rhythm_stats["true_positives"]
    false_negatives = rhythm_stats["false_negatives"]
    if true_positives + false_negatives == 0:
        return 0.0
    return true_positives / (true_positives + false_negatives)


def calculate_f1(rhythm_stats):
    precision = calculate_precision(rhythm_stats)
    recall = calculate_recall(rhythm_stats)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def get_rhythm(beatmap, passive=False):
    # Extract the rhythm from the beatmap
    # Active rhythm includes only circles, slider heads, and hold note heads
    # Passive rhythm also includes slider tails, slider repeats, and spinners tails
    rhythm = set()
    for hit_object in beatmap.hit_objects(stacking=False):
        if isinstance(hit_object, Circle):
            rhythm.add(int(hit_object.time.total_seconds() * 1000))
        elif isinstance(hit_object, Slider):
            duration: timedelta = (hit_object.end_time - hit_object.time) / hit_object.repeat
            rhythm.add(int(hit_object.time.total_seconds() * 1000))
            if passive:
                for i in range(hit_object.repeat):
                    rhythm.add(int((hit_object.time + duration * (i + 1)).total_seconds() * 1000))
        elif isinstance(hit_object, Spinner):
            if passive:
                rhythm.add(int(hit_object.end_time.total_seconds() * 1000))
        elif isinstance(hit_object, HoldNote):
            rhythm.add(int(hit_object.time.total_seconds() * 1000))

    return rhythm


def generate_beatmaps(beatmap_paths, fid_args: FidConfig, return_dict, idx):
    args = fid_args.inference
    args.device = fid_args.device
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')

    model, tokenizer, diff_model, diff_tokenizer, refine_model = None, None, None, None, None
    model, tokenizer = load_model(args.model_path, args.train, args.device, args.max_batch_size, True)

    if args.compile:
        model.transformer.forward = torch.compile(model.transformer.forward, mode="reduce-overhead", fullgraph=True)

    if args.generate_positions:
        diff_model, diff_tokenizer = load_diff_model(args.diff_ckpt, args.diffusion, args.device)

        if os.path.exists(args.diff_refine_ckpt):
            refine_model = load_diff_model(args.diff_refine_ckpt, args.diffusion, args.device)[0]

        if args.compile:
            diff_model.forward = torch.compile(diff_model.forward, mode="reduce-overhead", fullgraph=False)

    for beatmap_path in tqdm(beatmap_paths, desc=f"Process {idx}"):
        try:
            beatmap = Beatmap.from_path(beatmap_path)
            output_path = Path("generated") / beatmap_path.stem

            if fid_args.dataset_type == "ors":
                audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            else:
                audio_path = beatmap_path.parent / beatmap.audio_filename

            if fid_args.skip_generation or (output_path.exists() and len(list(output_path.glob("*.osu"))) > 0):
                if not output_path.exists() or len(list(output_path.glob("*.osu"))) == 0:
                    raise FileNotFoundError(f"Generated beatmap not found in {output_path}")
                print(f"Skipping {beatmap_path.stem} as it already exists")
            else:
                if ContextType.GD in args.in_context:
                    other_beatmaps = [k for k in beatmap_path.parent.glob("*.osu") if k != beatmap_path]
                    if len(other_beatmaps) == 0:
                        continue
                    other_beatmap_path = random.choice(other_beatmaps)
                else:
                    other_beatmap_path = beatmap_path

                generation_config = generation_config_from_beatmap(beatmap, tokenizer)
                beatmap_config = beatmap_config_from_beatmap(beatmap)
                beatmap_config.version = args.version

                if args.year is not None:
                    generation_config.year = args.year

                result = generate(
                    args,
                    audio_path=audio_path,
                    beatmap_path=other_beatmap_path,
                    output_path=output_path,
                    generation_config=generation_config,
                    beatmap_config=beatmap_config,
                    model=model,
                    tokenizer=tokenizer,
                    diff_model=diff_model,
                    diff_tokenizer=diff_tokenizer,
                    refine_model=refine_model,
                    verbose=False,
                )[0]
                generated_beatmap = Beatmap.parse(result)
                print(beatmap_path, "Generated %s hit objects" % len(generated_beatmap.hit_objects(stacking=False)))
        except Exception as e:
            print(f"Error processing {beatmap_path}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()  # Clear any cached memory


def calculate_metrics(args: FidConfig, beatmap_paths: list[Path]):
    print("Calculating metrics...")

    classifier_model, classifier_args, classifier_tokenizer = None, None, None
    if args.fid:
        classifier_model, classifier_args, classifier_tokenizer = load_ckpt(args.classifier_ckpt)

        if args.compile:
            classifier_model.model.transformer.forward = torch.compile(classifier_model.model.transformer.forward,
                                                                       mode="reduce-overhead", fullgraph=False)

    real_features = []
    generated_features = []
    active_rhythm_stats = {}
    passive_rhythm_stats = {}

    for beatmap_path in tqdm(beatmap_paths, desc=f"Metrics"):
        try:
            beatmap = Beatmap.from_path(beatmap_path)
            generated_path = Path("generated") / beatmap_path.stem

            if args.dataset_type == "ors":
                audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            else:
                audio_path = beatmap_path.parent / beatmap.audio_filename

            if generated_path.exists() and len(list(generated_path.glob("*.osu"))) > 0:
                generated_beatmap = Beatmap.from_path(list(generated_path.glob("*.osu"))[0])
            else:
                logger.warning(f"Skipping {beatmap_path.stem} as no generated beatmap found")
                continue

            if args.fid:
                # Calculate feature vectors for real and generated beatmaps
                sample_rate = classifier_args.data.sample_rate
                audio = load_audio_file(audio_path, sample_rate, normalize=args.inference.train.data.normalize_audio)

                for example in DataLoader(
                        ExampleDataset(beatmap, audio, classifier_args, classifier_tokenizer, args.device),
                        batch_size=args.classifier_batch_size):
                    classifier_result: OsuClassifierOutput = classifier_model(**example)
                    features = classifier_result.feature_vector
                    real_features.append(features.cpu().numpy())

                for example in DataLoader(
                        ExampleDataset(generated_beatmap, audio, classifier_args, classifier_tokenizer, args.device),
                        batch_size=args.classifier_batch_size):
                    classifier_result: OsuClassifierOutput = classifier_model(**example)
                    features = classifier_result.feature_vector
                    generated_features.append(features.cpu().numpy())

            if args.rhythm_stats:
                # Calculate rhythm stats
                real_active_rhythm = get_rhythm(beatmap, passive=False)
                generated_active_rhythm = get_rhythm(generated_beatmap, passive=False)
                add_to_dict(calculate_rhythm_stats(real_active_rhythm, generated_active_rhythm), active_rhythm_stats)

                real_passive_rhythm = get_rhythm(beatmap, passive=True)
                generated_passive_rhythm = get_rhythm(generated_beatmap, passive=True)
                add_to_dict(calculate_rhythm_stats(real_passive_rhythm, generated_passive_rhythm), passive_rhythm_stats)
        except Exception as e:
            print(f"Error processing {beatmap_path}: {e}")
            traceback.print_exc()
        finally:
            torch.cuda.empty_cache()  # Clear any cached memory

    if args.fid:
        # Calculate FID
        real_features = np.concatenate(real_features, axis=0)
        generated_features = np.concatenate(generated_features, axis=0)
        m1, s1 = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
        m2, s2 = np.mean(generated_features, axis=0), np.cov(generated_features, rowvar=False)
        fid = calculate_frechet_distance(m1, s1, m2, s2)

        logger.info(f"FID: {fid}")

    if args.rhythm_stats:
        # Calculate rhythm precision, recall, and F1 score
        active_precision = calculate_precision(active_rhythm_stats)
        active_recall = calculate_recall(active_rhythm_stats)
        active_f1 = calculate_f1(active_rhythm_stats)
        passive_precision = calculate_precision(passive_rhythm_stats)
        passive_recall = calculate_recall(passive_rhythm_stats)
        passive_f1 = calculate_f1(passive_rhythm_stats)
        logger.info(f"Active Rhythm Precision: {active_precision}")
        logger.info(f"Active Rhythm Recall: {active_recall}")
        logger.info(f"Active Rhythm F1: {active_f1}")
        logger.info(f"Passive Rhythm Precision: {passive_precision}")
        logger.info(f"Passive Rhythm Recall: {passive_recall}")
        logger.info(f"Passive Rhythm F1: {passive_f1}")


def test_training_set_overlap(beatmap_paths: list[Path], training_set_ids_path: Optional[str]):
    if training_set_ids_path is None:
        return

    if not os.path.exists(training_set_ids_path):
        logger.error(f"Training set IDs file {training_set_ids_path} does not exist.")
        return

    with open(training_set_ids_path, "r") as f:
        training_set_ids = set(int(line.strip()) for line in f)

    in_set = 0
    out_set = 0
    for path in tqdm(beatmap_paths):
        beatmap = Beatmap.from_path(path)
        if beatmap.beatmap_id in training_set_ids:
            in_set += 1
        else:
            out_set += 1
    logger.info(f"In training set: {in_set}, Not in training set: {out_set}, Total: {len(beatmap_paths)}, Ratio: {in_set / (in_set + out_set):.2f}")


@hydra.main(config_path="configs", config_name="calc_fid", version_base="1.1")
def main(args: FidConfig):
    prepare_args(args)

    # Fix inference model path
    if args.inference.model_path.startswith("./"):
        args.inference.model_path = os.path.join(Path(__file__).parent, args.inference.model_path[2:])

    beatmap_paths = get_beatmap_paths(args)

    test_training_set_overlap(beatmap_paths, args.training_set_ids_path)

    if not args.skip_generation:
        # Assign beatmaps to processes in a round-robin fashion
        num_processes = args.num_processes
        chunks = [[] for _ in range(num_processes)]
        for i, path in enumerate(beatmap_paths):
            chunks[i % num_processes].append(path)

        manager = Manager()
        return_dict = manager.dict()
        processes = []

        for i in range(num_processes):
            p = Process(target=generate_beatmaps, args=(chunks[i], args, return_dict, i))
            processes.append(p)
            p.start()

        for p in processes:
            p.join()

    calculate_metrics(args, beatmap_paths)


if __name__ == "__main__":
    main()
