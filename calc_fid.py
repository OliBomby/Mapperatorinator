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
from omegaconf import OmegaConf
from scipy import linalg
from slider import Beatmap, Circle, Slider, Spinner, HoldNote
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoProcessor, AutoModel

from classifier.classify import ExampleDataset
from classifier.libs.model.model import OsuClassifierOutput
from classifier.libs.utils import load_ckpt
from config import FidConfig, InferenceConfig
from inference import load_diff_model, generate, load_model_with_server, compile_device_and_seed, \
    setup_inference_environment
from osuT5.osuT5.dataset.data_utils import load_audio_file, load_mmrs_metadata, filter_mmrs_metadata
from osuT5.osuT5.inference import generation_config_from_beatmap, beatmap_config_from_beatmap
from osuT5.osuT5.tokenizer import ContextType
from multiprocessing import Process

# Add imports for multiprocessing-safe logging
import multiprocessing
from logging.handlers import QueueHandler, QueueListener

logger = logging.getLogger(__name__)


def _configure_generation_log_parent(log_file: Path) -> tuple[QueueListener, multiprocessing.Queue]:
    """Configure a QueueListener in the parent process that writes generation logs to a file."""
    log_file.parent.mkdir(parents=True, exist_ok=True)

    queue: multiprocessing.Queue = multiprocessing.Queue(-1)

    file_handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(
        logging.Formatter("[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] - %(message)s")
    )

    listener = QueueListener(queue, file_handler, respect_handler_level=True)
    listener.start()
    return listener, queue


def _configure_generation_log_worker(queue: multiprocessing.Queue) -> logging.Logger:
    """Configure the current process to send generation logs to the parent via QueueHandler."""
    if queue is None:
        # Fallback: no queue provided (e.g., single-process mode). Log to a local file.
        gen_logger = logging.getLogger("calc_fid.generation")
        gen_logger.setLevel(logging.INFO)
        if not any(isinstance(h, logging.FileHandler) and getattr(h, "baseFilename", "").endswith("generation.log")
                   for h in gen_logger.handlers):
            fh = logging.FileHandler("generation.log", mode="a", encoding="utf-8")
            fh.setFormatter(
                logging.Formatter("[%(asctime)s][%(processName)s][%(name)s][%(levelname)s] - %(message)s")
            )
            gen_logger.addHandler(fh)
        gen_logger.propagate = False
        return gen_logger

    gen_logger = logging.getLogger("calc_fid.generation")
    gen_logger.setLevel(logging.INFO)

    # Avoid duplicates if this function is called more than once in the same process.
    if not any(isinstance(h, QueueHandler) for h in gen_logger.handlers):
        gen_logger.addHandler(QueueHandler(queue))

    # Prevent propagation into Hydra/root handlers (keeps calc_fid.log clean).
    gen_logger.propagate = False
    return gen_logger


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
        beatmap_files = [dataset_path / "data" / item["BeatmapSetFolder"] / item["BeatmapFile"] for _, item in
                         filtered_metadata.iterrows()]
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
            rhythm.add(int(hit_object.time.total_seconds() * 1000 + 1e-5))
        elif isinstance(hit_object, Slider):
            duration: timedelta = (hit_object.end_time - hit_object.time) / hit_object.repeat
            rhythm.add(int(hit_object.time.total_seconds() * 1000 + 1e-5))
            if passive:
                for i in range(hit_object.repeat):
                    rhythm.add(int((hit_object.time + duration * (i + 1)).total_seconds() * 1000 + 1e-5))
        elif isinstance(hit_object, Spinner):
            if passive:
                rhythm.add(int(hit_object.end_time.total_seconds() * 1000 + 1e-5))
        elif isinstance(hit_object, HoldNote):
            rhythm.add(int(hit_object.time.total_seconds() * 1000 + 1e-5))

    return rhythm


def generate_beatmaps(beatmap_paths, args: InferenceConfig, dataset_type, idx, log_queue=None):
    torch.set_grad_enabled(False)
    torch.set_float32_matmul_precision('high')

    gen_logger = _configure_generation_log_worker(log_queue)

    model, tokenizer, diff_model, diff_tokenizer, refine_model = None, None, None, None, None
    model, tokenizer = load_model_with_server(
        args.model_path,
        args.train,
        args.device,
        max_batch_size=args.max_batch_size,
        use_server=args.use_server,
        precision=args.precision,
        attn_implementation=args.attn_implementation,
    )

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

            if dataset_type == "ors":
                audio_path = beatmap_path.parents[1] / list(beatmap_path.parents[1].glob('audio.*'))[0]
            else:
                audio_path = beatmap_path.parent / beatmap.audio_filename

            if output_path.exists() and len(list(output_path.glob("*.osu"))) > 0:
                if not output_path.exists() or len(list(output_path.glob("*.osu"))) == 0:
                    raise FileNotFoundError(f"Generated beatmap not found in {output_path}")
                gen_logger.info("Skipping %s as it already exists", beatmap_path.stem)
            else:
                if ContextType.GD in args.in_context:
                    other_beatmaps = [k for k in beatmap_path.parent.glob("*.osu") if k != beatmap_path]
                    if len(other_beatmaps) == 0:
                        continue
                    other_beatmap_path = random.choice(other_beatmaps)
                else:
                    other_beatmap_path = beatmap_path

                generation_config = generation_config_from_beatmap(beatmap, beatmap_path, tokenizer)
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
                    logger=gen_logger,
                )[0]
                generated_beatmap = Beatmap.parse(result)
                gen_logger.info(
                    "%s Generated %s hit objects",
                    str(beatmap_path),
                    len(generated_beatmap.hit_objects(stacking=False)),
                )
        except Exception:
            gen_logger.exception("Error processing %s", beatmap_path)
        finally:
            torch.cuda.empty_cache()  # Clear any cached memory


@torch.no_grad()
def calculate_metrics(args: FidConfig, beatmap_paths: list[Path]):
    print("Calculating metrics...")

    classifier_model, classifier_args, classifier_tokenizer = None, None, None
    if args.fid:
        classifier_model, classifier_args, classifier_tokenizer = load_ckpt(args.classifier_ckpt)

        if args.compile:
            classifier_model.model.transformer.forward = torch.compile(classifier_model.model.transformer.forward,
                                                                       mode="reduce-overhead", fullgraph=False)

    cm3p_model, cm3p_processor = None, None
    if args.fid_cm3p:
        cm3p_processor = AutoProcessor.from_pretrained(args.cm3p_ckpt, trust_remote_code=True, revision="main")
        cm3p_model = AutoModel.from_pretrained(args.cm3p_ckpt, device_map=args.device, dtype=torch.bfloat16,
                                               trust_remote_code=True, revision="main")

    real_features = []
    generated_features = []
    real_features_cm3p = []
    generated_features_cm3p = []
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
                audio = load_audio_file(audio_path, sample_rate)

                def process(process_beatmap, feature_list):
                    for example in DataLoader(
                            ExampleDataset(process_beatmap, audio, classifier_args, classifier_tokenizer, args.device),
                            batch_size=args.classifier_batch_size):
                        classifier_result: OsuClassifierOutput = classifier_model(**example)
                        features = classifier_result.feature_vector
                        feature_list.append(features.cpu().numpy())

                process(beatmap, real_features)
                process(generated_beatmap, generated_features)

            if args.fid_cm3p:
                def process(process_beatmap, feature_list):
                    beatmap_data = cm3p_processor(beatmap=process_beatmap, audio=audio_path)
                    beatmap_data = beatmap_data.to(args.device, dtype=torch.bfloat16)
                    # Turn dict of tensors into list of dicts of tensors for DataLoader
                    beatmap_data = [{key: beatmap_data[key][i] for key in beatmap_data} for i in
                                    range(len(beatmap_data['input_ids']))]
                    for example in DataLoader(beatmap_data, batch_size=args.cm3p_batch_size):
                        outputs = cm3p_model(**example, return_loss=False)
                        beatmap_embeds = outputs.beatmap_embeds
                        feature_list.append(beatmap_embeds.float().cpu().numpy())

                process(beatmap, real_features_cm3p)
                process(generated_beatmap, generated_features_cm3p)

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

    def fid_calc(features1, features2, name):
        features1 = np.concatenate(features1, axis=0)
        features2 = np.concatenate(features2, axis=0)
        m1, s1 = np.mean(features1, axis=0), np.cov(features1, rowvar=False)
        m2, s2 = np.mean(features2, axis=0), np.cov(features2, rowvar=False)
        fid = calculate_frechet_distance(m1, s1, m2, s2)
        logger.info(f"{name}: {fid}")

    if args.fid:
        fid_calc(real_features, generated_features, "FID")

    if args.fid_cm3p:
        fid_calc(real_features_cm3p, generated_features_cm3p, "FID CM3P")

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
    logger.info(
        f"In training set: {in_set}, Not in training set: {out_set}, Total: {len(beatmap_paths)}, Ratio: {in_set / (in_set + out_set):.2f}")


@hydra.main(config_path="configs", config_name="calc_fid", version_base="1.1")
def main(args: FidConfig):
    args: FidConfig = OmegaConf.to_object(args)
    compile_device_and_seed(args.inference)
    setup_inference_environment(args.inference.seed)
    args.device = args.inference.device

    print(f"Logging to directory: {os.getcwd()}")

    # Fix inference model path
    if args.inference.model_path.startswith("./"):
        args.inference.model_path = os.path.join(Path(__file__).parent, args.inference.model_path[2:])

    beatmap_paths = get_beatmap_paths(args)

    test_training_set_overlap(beatmap_paths, args.training_set_ids_path)

    listener = None
    try:
        # Configure generation logger (writes to generation.log in the Hydra run dir)
        listener, log_queue = _configure_generation_log_parent(Path(os.getcwd()) / "generation.log")

        if not args.skip_generation:
            # Assign beatmaps to processes in a round-robin fashion
            num_processes = max(args.num_processes, 1)
            chunks = [[] for _ in range(num_processes)]
            for i, path in enumerate(beatmap_paths):
                chunks[i % num_processes].append(path)

            if args.num_processes <= 0:
                generate_beatmaps(chunks[0], args.inference, args.dataset_type, 0, log_queue=log_queue)
            else:
                processes = []
                for i in range(num_processes):
                    p = Process(target=generate_beatmaps, args=(chunks[i], args.inference, args.dataset_type, i, log_queue))
                    processes.append(p)
                    p.start()

                for p in processes:
                    p.join()

        calculate_metrics(args, beatmap_paths)
    finally:
        if listener is not None:
            listener.stop()


if __name__ == "__main__":
    main()
