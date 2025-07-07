import hydra
import pandas as pd

from config import InferenceConfig
from inference import load_model
from osuT5.osuT5.event import EventType
from osuT5.osuT5.model import Mapperatorinator
from osuT5.osuT5.tokenizer import Tokenizer


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


@hydra.main(config_path="configs", config_name="inference_v31", version_base="1.1")
def main(args: InferenceConfig):
    model_name = "OliBomby/Mapperatorinator-v31"

    model, tokenizer = load_model(args.model_path, args.train, args.device)

    # Remove mappers from removed_users.csv
    with open("datasets/removed_users.txt", 'r') as f:
        removed_users = [int(line.strip()) for line in f if line.strip()]
    remove_mappers_from_model(model, tokenizer, removed_users)

    model.push_to_hub(model_name, private=True)
    tokenizer.push_to_hub(model_name, private=True)

    model = Mapperatorinator.from_pretrained(model_name)
    tokenizer = Tokenizer.from_pretrained(model_name)

    print("Done")

if __name__ == "__main__":
    main()
