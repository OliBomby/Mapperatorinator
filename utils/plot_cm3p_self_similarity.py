import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter
import torch
import torch.nn.functional as F
from slider import Beatmap
from tqdm import tqdm
from transformers import AutoModel, AutoProcessor


DEFAULT_CM3P_CKPT = "OliBomby/CM3P"


@dataclass
class BeatmapAnalysis:
    beatmap_path: Path
    audio_path: Path
    features: np.ndarray
    window_starts: np.ndarray
    matrix: np.ndarray
    overall_self_similarity: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute and plot a CM3P self-similarity matrix for a beatmap.",
    )
    parser.add_argument("beatmap_path", type=Path, help="Path to the .osu file.")
    parser.add_argument(
        "--beatmap-path-2",
        type=Path,
        default=None,
        help="Optional second .osu file to analyze and compare against the first.",
    )
    parser.add_argument(
        "--audio-path",
        type=Path,
        default=None,
        help="Optional path to the audio file. If omitted, it is inferred from the beatmap.",
    )
    parser.add_argument(
        "--audio-path-2",
        type=Path,
        default=None,
        help="Optional audio path for the second beatmap. If omitted, it is inferred from that beatmap.",
    )
    parser.add_argument(
        "--cm3p-ckpt",
        type=str,
        default=DEFAULT_CM3P_CKPT,
        help=f"CM3P checkpoint to load (default: {DEFAULT_CM3P_CKPT}).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use: auto, cpu, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["auto", "fp32", "bf16", "fp16"],
        default="auto",
        help="Model dtype to use.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for CM3P embedding inference.",
    )
    parser.add_argument(
        "--window-length-sec",
        type=float,
        default=None,
        help="CM3P window length in seconds. Defaults to the processor setting.",
    )
    parser.add_argument(
        "--window-stride-sec",
        type=float,
        default=None,
        help="CM3P window stride in seconds. Defaults to the processor setting.",
    )
    parser.add_argument(
        "--min-window-length-sec",
        type=float,
        default=None,
        help="Minimum trailing window length before CM3P stops creating windows. Defaults to the processor setting.",
    )
    parser.add_argument(
        "--similarity",
        type=str,
        choices=["cosine", "dot"],
        default="cosine",
        help="Similarity metric for the self-similarity matrix.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output image path. If omitted, the plot is shown only and not saved.",
    )
    parser.add_argument(
        "--save-npz",
        type=Path,
        default=None,
        help="Optional .npz path to save embeddings, window starts, and the similarity matrix.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default=None,
        help="Optional plot title override.",
    )
    parser.add_argument(
        "--cmap",
        type=str,
        default="magma",
        help="Matplotlib colormap for the heatmap.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Do not open an interactive plot window.",
    )
    return parser.parse_args()


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    if dtype == "fp32":
        return torch.float32
    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16

    if device.startswith("cuda") and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def find_audio_path(beatmap_path: Path, beatmap: Beatmap) -> Path:
    candidates: list[Path] = []

    if beatmap.audio_filename:
        candidates.append(beatmap_path.parent / beatmap.audio_filename)

    candidates.extend(sorted(beatmap_path.parent.glob("audio.*")))

    if beatmap_path.parent.name.lower() == "beatmaps" and len(beatmap_path.parents) >= 2:
        candidates.extend(sorted(beatmap_path.parents[1].glob("audio.*")))

    seen = set()
    unique_candidates = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in seen:
            seen.add(resolved)
            unique_candidates.append(candidate)

    for candidate in unique_candidates:
        if candidate.exists():
            return candidate

    searched = "\n - ".join(str(path) for path in unique_candidates) if unique_candidates else "<none>"
    raise FileNotFoundError(
        "Could not infer the audio file for the beatmap. Checked:\n"
        f" - {searched}"
    )


def compute_window_starts(song_length: float, window_stride_sec: float, min_window_length_sec: float) -> np.ndarray:
    if song_length <= min_window_length_sec:
        return np.zeros((0,), dtype=np.float32)
    return np.arange(0.0, song_length - min_window_length_sec, window_stride_sec, dtype=np.float32)


def infer_song_length_seconds(processor, audio_path: Path) -> float:
    sampling_rate = processor.default_kwargs["audio_kwargs"]["sampling_rate"]
    audio = processor._load_audio(sampling_rate, audio_path, audio_sampling_rate=None)
    if len(audio) != 1:
        raise RuntimeError(f"Expected exactly one audio item, got {len(audio)}")
    return len(audio[0]) / sampling_rate


def extract_window_embeddings(
    processor,
    model,
    beatmap: Beatmap,
    audio_path: Path,
    device: str,
    model_dtype: torch.dtype,
    batch_size: int,
    window_length_sec: float,
    window_stride_sec: float,
    min_window_length_sec: float,
) -> tuple[np.ndarray, np.ndarray]:
    beatmap_data = processor(
        beatmap=beatmap,
        audio=audio_path,
        window_length_sec=window_length_sec,
        window_stride_sec=window_stride_sec,
        min_window_length_sec=min_window_length_sec,
    )

    num_windows = int(beatmap_data["input_ids"].shape[0])
    if num_windows == 0:
        raise ValueError("CM3P did not produce any windows for this beatmap.")

    song_length = infer_song_length_seconds(processor, audio_path)
    window_starts = compute_window_starts(song_length, window_stride_sec, min_window_length_sec)
    if len(window_starts) != num_windows:
        window_starts = window_starts[:num_windows]
        if len(window_starts) != num_windows:
            raise RuntimeError(
                f"Window start count mismatch: expected {num_windows}, computed {len(window_starts)}"
            )

    feature_batches = []
    for start in tqdm(range(0, num_windows, batch_size), desc="Embedding windows"):
        end = min(start + batch_size, num_windows)
        batch = {}
        for key, value in beatmap_data.items():
            batch_value = value[start:end]
            if torch.is_tensor(batch_value):
                if batch_value.is_floating_point():
                    batch_value = batch_value.to(device=device, dtype=model_dtype)
                else:
                    batch_value = batch_value.to(device=device)
            batch[key] = batch_value

        outputs = model(**batch, return_loss=False)
        feature_batches.append(outputs.beatmap_embeds.float().cpu())

    features = torch.cat(feature_batches, dim=0).numpy()
    return features, window_starts


def compute_self_similarity(features: np.ndarray, similarity: str) -> np.ndarray:
    if similarity == "dot":
        return features @ features.T

    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms = np.clip(norms, a_min=1e-12, a_max=None)
    normalized = features / norms
    return normalized @ normalized.T


def normalize_matrix_for_display(
    matrix: np.ndarray,
    similarity: str,
    min_value: float | None = None,
    max_value: float | None = None,
) -> np.ndarray:
    if similarity == "cosine":
        return np.clip((matrix + 1.0) / 2.0, 0.0, 1.0)

    if min_value is None:
        min_value = float(np.min(matrix))
    if max_value is None:
        max_value = float(np.max(matrix))

    if max_value - min_value < 1e-12:
        return np.zeros_like(matrix, dtype=np.float32)

    return np.clip((matrix - min_value) / (max_value - min_value), 0.0, 1.0).astype(np.float32)


def compute_overall_self_similarity(matrix: np.ndarray, similarity: str) -> float:
    display_matrix = normalize_matrix_for_display(matrix, similarity)
    if display_matrix.ndim != 2 or display_matrix.shape[0] != display_matrix.shape[1]:
        raise ValueError("Self-similarity matrix must be square.")

    if display_matrix.shape[0] <= 1:
        return 0.0

    off_diagonal_mask = ~np.eye(display_matrix.shape[0], dtype=bool)
    off_diagonal_values = display_matrix[off_diagonal_mask]
    if off_diagonal_values.size == 0:
        return 0.0

    return float(np.mean(off_diagonal_values))


def resize_matrix(matrix: np.ndarray, target_size: int) -> np.ndarray:
    if matrix.shape == (target_size, target_size):
        return matrix.astype(np.float32, copy=False)

    matrix_tensor = torch.from_numpy(matrix.astype(np.float32, copy=False)).unsqueeze(0).unsqueeze(0)
    resized = F.interpolate(matrix_tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy()


def compute_ssim(image_a: np.ndarray, image_b: np.ndarray, data_range: float = 1.0) -> float:
    image_a = image_a.astype(np.float64, copy=False)
    image_b = image_b.astype(np.float64, copy=False)

    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2

    mu_a = gaussian_filter(image_a, sigma=1.5)
    mu_b = gaussian_filter(image_b, sigma=1.5)

    mu_a_sq = mu_a * mu_a
    mu_b_sq = mu_b * mu_b
    mu_ab = mu_a * mu_b

    sigma_a_sq = gaussian_filter(image_a * image_a, sigma=1.5) - mu_a_sq
    sigma_b_sq = gaussian_filter(image_b * image_b, sigma=1.5) - mu_b_sq
    sigma_ab = gaussian_filter(image_a * image_b, sigma=1.5) - mu_ab

    ssim_numerator = (2.0 * mu_ab + c1) * (2.0 * sigma_ab + c2)
    ssim_denominator = (mu_a_sq + mu_b_sq + c1) * (sigma_a_sq + sigma_b_sq + c2)
    ssim_map = ssim_numerator / np.maximum(ssim_denominator, 1e-12)
    return float(np.mean(ssim_map))


def compare_self_similarity_matrices(
    matrix_a: np.ndarray,
    matrix_b: np.ndarray,
    similarity: str,
) -> tuple[float, float, int]:
    if similarity == "dot":
        shared_min = float(min(np.min(matrix_a), np.min(matrix_b)))
        shared_max = float(max(np.max(matrix_a), np.max(matrix_b)))
        display_a = normalize_matrix_for_display(matrix_a, similarity, shared_min, shared_max)
        display_b = normalize_matrix_for_display(matrix_b, similarity, shared_min, shared_max)
    else:
        display_a = normalize_matrix_for_display(matrix_a, similarity)
        display_b = normalize_matrix_for_display(matrix_b, similarity)

    target_size = max(display_a.shape[0], display_b.shape[0])
    resized_a = resize_matrix(display_a, target_size)
    resized_b = resize_matrix(display_b, target_size)

    mse = float(np.mean((resized_a - resized_b) ** 2))
    ssim = compute_ssim(resized_a, resized_b, data_range=1.0)
    return mse, ssim, target_size


def format_seconds(seconds: float) -> str:
    total_seconds = max(int(round(seconds)), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:d}:{secs:02d}"


def _apply_axis_labels(ax, window_starts: np.ndarray, window_length_sec: float) -> None:
    tick_count = min(12, len(window_starts))
    if tick_count > 0:
        tick_positions = np.linspace(0, len(window_starts) - 1, num=tick_count, dtype=int)
        tick_labels = [format_seconds(float(window_starts[idx])) for idx in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha="right")
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels)

    ax.set_xlabel(f"Window start time ({window_length_sec:.2f}s windows)")
    ax.set_ylabel(f"Window start time ({window_length_sec:.2f}s windows)")


def plot_similarity_matrices(
    analyses: list[BeatmapAnalysis],
    window_length_sec: float,
    output_path: Path | None,
    cmap: str,
    similarity: str,
    title: str | None,
    show: bool,
) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, len(analyses), figsize=(10 * len(analyses), 8), squeeze=False)
    axes = axes[0]

    shared_vmin = min(float(np.min(analysis.matrix)) for analysis in analyses)
    shared_vmax = max(float(np.max(analysis.matrix)) for analysis in analyses)

    for ax, analysis in zip(axes, analyses):
        imshow_kwargs: dict[str, object] = {"cmap": cmap, "origin": "lower", "interpolation": "nearest"}
        if shared_vmin is not None and shared_vmax is not None:
            imshow_kwargs.update({"vmin": shared_vmin, "vmax": shared_vmax})

        image = ax.imshow(analysis.matrix, **imshow_kwargs)
        cbar = fig.colorbar(image, ax=ax)
        cbar.set_label(f"{similarity.title()} similarity")

        _apply_axis_labels(ax, analysis.window_starts, window_length_sec)
        ax.set_title(
            f"{analysis.beatmap_path.name}\n"
            f"overall self-similarity = {analysis.overall_self_similarity:.6f}"
        )

    if title:
        fig.suptitle(title)
    fig.tight_layout()
    if title:
        fig.subplots_adjust(top=0.88)

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close(fig)


def analyze_beatmap(
    beatmap_path: Path,
    audio_path: Path | None,
    processor,
    model,
    device: str,
    model_dtype: torch.dtype,
    batch_size: int,
    window_length_sec: float,
    window_stride_sec: float,
    min_window_length_sec: float,
    similarity: str,
) -> BeatmapAnalysis:
    beatmap = Beatmap.from_path(beatmap_path)
    resolved_audio_path = audio_path.resolve() if audio_path is not None else find_audio_path(beatmap_path, beatmap)
    if not resolved_audio_path.exists():
        raise FileNotFoundError(f"Audio path does not exist: {resolved_audio_path}")

    with torch.inference_mode():
        features, window_starts = extract_window_embeddings(
            processor=processor,
            model=model,
            beatmap=beatmap,
            audio_path=resolved_audio_path,
            device=device,
            model_dtype=model_dtype,
            batch_size=batch_size,
            window_length_sec=window_length_sec,
            window_stride_sec=window_stride_sec,
            min_window_length_sec=min_window_length_sec,
        )

    matrix = compute_self_similarity(features, similarity)
    overall_self_similarity = compute_overall_self_similarity(matrix, similarity)

    return BeatmapAnalysis(
        beatmap_path=beatmap_path,
        audio_path=resolved_audio_path,
        features=features,
        window_starts=window_starts,
        matrix=matrix,
        overall_self_similarity=overall_self_similarity,
    )


def main() -> None:
    args = parse_args()

    beatmap_path = args.beatmap_path.resolve()
    if not beatmap_path.exists():
        raise FileNotFoundError(f"Beatmap path does not exist: {beatmap_path}")

    beatmap_path_2 = args.beatmap_path_2.resolve() if args.beatmap_path_2 is not None else None
    if beatmap_path_2 is not None and not beatmap_path_2.exists():
        raise FileNotFoundError(f"Second beatmap path does not exist: {beatmap_path_2}")

    if args.batch_size <= 0:
        raise ValueError("Batch size must be positive.")

    if args.no_show:
        import matplotlib

        matplotlib.use("Agg")

    device = resolve_device(args.device)
    model_dtype = resolve_dtype(args.dtype, device)

    print(f"Loading CM3P processor from {args.cm3p_ckpt}...")
    processor = AutoProcessor.from_pretrained(args.cm3p_ckpt, trust_remote_code=True, revision="main")

    processor_beatmap_kwargs = processor.default_kwargs["beatmap_kwargs"]
    window_length_sec = (
        args.window_length_sec
        if args.window_length_sec is not None
        else float(processor_beatmap_kwargs["window_length_sec"])
    )
    window_stride_sec = (
        args.window_stride_sec
        if args.window_stride_sec is not None
        else float(processor_beatmap_kwargs["window_stride_sec"])
    )
    min_window_length_sec = (
        args.min_window_length_sec
        if args.min_window_length_sec is not None
        else float(processor_beatmap_kwargs.get("min_window_length_sec", 1.0))
    )

    if window_length_sec <= 0 or window_stride_sec <= 0 or min_window_length_sec <= 0:
        raise ValueError("Window length, stride, and minimum window length must all be positive.")

    print(
        "Using CM3P windows: "
        f"length={window_length_sec:.2f}s, stride={window_stride_sec:.2f}s, "
        f"min_length={min_window_length_sec:.2f}s"
    )

    print(f"Loading CM3P model on {device} with dtype={model_dtype}...")
    model = AutoModel.from_pretrained(
        args.cm3p_ckpt,
        trust_remote_code=True,
        revision="main",
        dtype=model_dtype,
    )
    model = model.to(device)
    model.eval()

    analyses = [
        analyze_beatmap(
            beatmap_path=beatmap_path,
            audio_path=args.audio_path,
            processor=processor,
            model=model,
            device=device,
            model_dtype=model_dtype,
            batch_size=args.batch_size,
            window_length_sec=window_length_sec,
            window_stride_sec=window_stride_sec,
            min_window_length_sec=min_window_length_sec,
            similarity=args.similarity,
        )
    ]

    if beatmap_path_2 is not None:
        analyses.append(
            analyze_beatmap(
                beatmap_path=beatmap_path_2,
                audio_path=args.audio_path_2,
                processor=processor,
                model=model,
                device=device,
                model_dtype=model_dtype,
                batch_size=args.batch_size,
                window_length_sec=window_length_sec,
                window_stride_sec=window_stride_sec,
                min_window_length_sec=min_window_length_sec,
                similarity=args.similarity,
            )
        )

    for index, analysis in enumerate(analyses, start=1):
        suffix = "" if len(analyses) == 1 else f" {index}"
        print(f"Beatmap{suffix}: {analysis.beatmap_path}")
        print(f"Audio{suffix}: {analysis.audio_path}")
        print(f"CM3P windows{suffix}: {len(analysis.window_starts)}")
        print(f"Overall self-similarity{suffix}: {analysis.overall_self_similarity:.6f}")

    comparison_metrics = None
    if len(analyses) == 2:
        mse, ssim, comparison_size = compare_self_similarity_matrices(
            analyses[0].matrix,
            analyses[1].matrix,
            args.similarity,
        )
        comparison_metrics = (mse, ssim, comparison_size)
        print(f"SSM comparison resize: {comparison_size}x{comparison_size}")
        print(f"SSM MSE: {mse:.6f}")
        print(f"SSM SSIM: {ssim:.6f}")

    plot_similarity_matrices(
        analyses=analyses,
        window_length_sec=window_length_sec,
        output_path=args.output,
        cmap=args.cmap,
        similarity=args.similarity,
        title=args.title,
        show=not args.no_show,
    )

    if args.save_npz is not None:
        args.save_npz.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, object] = {
            "window_length_sec": np.array(window_length_sec, dtype=np.float32),
            "window_stride_sec": np.array(window_stride_sec, dtype=np.float32),
            "min_window_length_sec": np.array(min_window_length_sec, dtype=np.float32),
        }
        for index, analysis in enumerate(analyses, start=1):
            suffix = "" if index == 1 else f"_{index}"
            payload[f"beatmap_path{suffix}"] = str(analysis.beatmap_path)
            payload[f"audio_path{suffix}"] = str(analysis.audio_path)
            payload[f"features{suffix}"] = analysis.features
            payload[f"window_starts_sec{suffix}"] = analysis.window_starts
            payload[f"similarity{suffix}"] = analysis.matrix
            payload[f"overall_self_similarity{suffix}"] = np.array(analysis.overall_self_similarity, dtype=np.float32)

        if comparison_metrics is not None:
            mse, ssim, comparison_size = comparison_metrics
            payload["ssm_mse"] = np.array(mse, dtype=np.float32)
            payload["ssm_ssim"] = np.array(ssim, dtype=np.float32)
            payload["comparison_size"] = np.array(comparison_size, dtype=np.int32)

        np.savez_compressed(args.save_npz, **payload)
        print(f"Saved features and matrix to {args.save_npz}")

    if args.output is not None:
        print(f"Saved plot to {args.output}")
    elif args.no_show:
        print("Plot was not shown or saved.")
    else:
        print("Displayed plot without saving it.")


if __name__ == "__main__":
    main()



