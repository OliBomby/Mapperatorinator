import gc
from pathlib import Path
from tempfile import TemporaryDirectory

import hydra
from huggingface_hub import HfApi
from omegaconf import OmegaConf, DictConfig

from config import InferenceConfig
from inference import load_model_with_server
from osuT5.osuT5.event import EventType
from osuT5.osuT5.model import Mapperatorinator
from osuT5.osuT5.tokenizer import Tokenizer


MODEL_WEIGHT_PATTERNS = ("model*.safetensors", "pytorch_model*.bin")
TOKENIZER_STATE_FILENAMES = ("tokenizer.json", "custom_checkpoint_0.pkl")


def remove_mappers_from_model(model, tokenizer, removed_users: list[int]):
    if not hasattr(tokenizer, "mapper_idx"):
        print("Tokenizer does not have mapper_idx, nothing to remove.")
        return

    # Null any mapper embeddings
    if hasattr(model, "mapper_embedder"):
        for user in removed_users:
            if user in tokenizer.mapper_idx:
                user_idx = tokenizer.mapper_idx.get(user)
                model.mapper_embedder.embedding.weight.data[user_idx].zero_()
                print(f"Nulled idx {user_idx} ({user}) in mapper embedder.")

    # Null any mapper token embeddings
    if EventType.MAPPER in tokenizer.event_range and hasattr(model, "decoder_embedder"):
        for user in removed_users:
            if user in tokenizer.mapper_idx:
                user_token_idx = tokenizer.encode_mapper_id(user)
                model.decoder_embedder.weight.data[user_token_idx].zero_()
                print(f"Nulled idx {user_token_idx} ({user}) in decoder embedder.")

    # Remove mapper from the idx mapping
    if hasattr(tokenizer, "mapper_idx"):
        for user in removed_users:
            if user in tokenizer.mapper_idx:
                del tokenizer.mapper_idx[user]
                print(f"Removed mapper {user} from tokenizer idx mapping.")


def load_removed_users() -> list[int]:
    removed_users_path = Path(__file__).resolve().parent / "datasets" / "removed_users.txt"
    with open(removed_users_path, "r", encoding="utf-8") as f:
        return [int(line.strip()) for line in f if line.strip()]


def contains_loadable_model_checkpoint(path: Path) -> bool:
    has_model_weights = any(
        any(path.glob(pattern))
        for pattern in MODEL_WEIGHT_PATTERNS
    )
    has_tokenizer_state = any((path / filename).is_file() for filename in TOKENIZER_STATE_FILENAMES)
    return has_model_weights and has_tokenizer_state


def discover_submodel_paths(model_path: str | Path) -> list[Path]:
    model_root = Path(model_path)
    if not model_root.is_dir():
        print(f"Model path {model_path} is not a local directory, skipping subfolder discovery.")
        return []

    discovered_paths = [
        path
        for path in sorted(model_root.rglob("*"))
        if path.is_dir() and contains_loadable_model_checkpoint(path)
    ]
    print(f"Discovered {len(discovered_paths)} submodel folder(s).")
    for path in discovered_paths:
        print(f" - {path.relative_to(model_root).as_posix()}")
    return discovered_paths


def load_local_model_checkpoint(model_path: str | Path, train_args):
    print(f"Loading model checkpoint from {model_path}")
    return load_model_with_server(
        model_path,
        train_args,
        "cpu",
        use_server=False,
        gamemode=None,
        auto_select_gamemode_model=False,
    )


def save_model_bundle(model, tokenizer, save_directory: Path):
    save_directory.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(save_directory, safe_serialization=True)
    tokenizer.save_pretrained(save_directory)

    missing_files = [
        filename
        for filename in ("config.json", "generation_config.json", "tokenizer.json")
        if not (save_directory / filename).is_file()
    ]
    if not any(save_directory.glob("model*.safetensors")):
        missing_files.append("model*.safetensors")

    if missing_files:
        raise FileNotFoundError(
            f"Missing expected saved artifacts in {save_directory}: {', '.join(missing_files)}"
        )

    print(f"Saved artifacts to {save_directory}")


def verify_uploaded_checkpoint(repo_id: str, subfolder: str = ""):
    location = repo_id if not subfolder else f"{repo_id}/{subfolder}"
    print(f"Verifying uploaded checkpoint at {location}")
    model = Mapperatorinator.from_pretrained(repo_id, subfolder=subfolder, device_map="cpu")
    tokenizer = Tokenizer.from_pretrained(repo_id, subfolder=subfolder)
    del model, tokenizer
    gc.collect()
    print(f"Verified loading {location}")


@hydra.main(config_path="configs/inference", config_name="v32", version_base="1.1")
def main(args: InferenceConfig):
    args = OmegaConf.to_object(args) if isinstance(args, DictConfig) else args
    model_name = "OliBomby/Mapperatorinator-v32"

    removed_users = load_removed_users()
    model_root = Path(args.model_path)
    submodel_paths = discover_submodel_paths(model_root)
    checkpoints_to_upload = [(None, args.model_path)] + [
        (path.relative_to(model_root).as_posix(), path)
        for path in submodel_paths
    ]

    api = HfApi()
    api.create_repo(repo_id=model_name, repo_type="model", private=True, exist_ok=True)

    with TemporaryDirectory(prefix="push_to_hub_") as temp_dir:
        staged_root = Path(temp_dir)

        for relative_subfolder, source_path in checkpoints_to_upload:
            model, tokenizer = load_local_model_checkpoint(source_path, args.train)
            remove_mappers_from_model(model, tokenizer, removed_users)

            save_directory = staged_root if relative_subfolder is None else staged_root / relative_subfolder
            save_model_bundle(model, tokenizer, save_directory)

            del model, tokenizer
            gc.collect()

        api.upload_folder(
            repo_id=model_name,
            repo_type="model",
            folder_path=str(staged_root),
            commit_message="Upload root and subfolder checkpoints",
        )

    verify_uploaded_checkpoint(model_name)
    for relative_subfolder, _ in checkpoints_to_upload[1:]:
        verify_uploaded_checkpoint(model_name, relative_subfolder)

    print("Done")

if __name__ == "__main__":
    main()
