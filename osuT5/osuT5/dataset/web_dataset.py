from typing import Optional

from huggingface_hub import list_repo_files
from datasets import load_dataset, Audio, IterableDataset

from ..config import DataConfig
from ..dataset.osu_parser import OsuParser
from ..tokenizer import Tokenizer


def get_web_dataset(
            args: DataConfig,
            parser: OsuParser,
            tokenizer: Tokenizer,
            subset_ids: Optional[list[int]] = None,
            test: bool = False,
            **_
    ) -> IterableDataset:
        repo_id = args.test_dataset_path if test else args.train_dataset_path
        dataset_start = args.test_dataset_start if test else args.train_dataset_start
        dataset_end = args.test_dataset_end if test else args.train_dataset_end

        all_files = [f for f in list_repo_files(repo_id, repo_type="dataset") if f.startswith("compressed/")]
        all_files.sort() # Ensure consistent order

        files_split = all_files[dataset_start:dataset_end]

        dataset = load_dataset(repo_id, data_files=files_split, streaming=True, split="train")

        print(f"Num shards: {len(files_split)}")

        dataset = dataset.cast_column("opus", Audio(sampling_rate=args.sample_rate, num_channels=1))
        dataset = dataset.shuffle(seed=42, buffer_size=1)

        print(dataset)

        return dataset
        # print(dataset.features)
        # for idx, example in enumerate(dataset["__key__"]):
        #     print(example)
        #     if idx == 2:
        #         state_dict = dataset.state_dict()
        #         print("checkpoint")
        #         break
        # dataset.load_state_dict(state_dict)
        # print(f"restart from checkpoint")
        # for idx, example in enumerate(dataset["__key__"]):
        #     print(example)
        #     if idx == 2:
        #         break


        # train_ds = train_ds.filter(lambda x: int(x['approved_date'].split("-")[0]) >= 2018)

        # loader = DataLoader(windowed_train_ds, batch_size=32, num_workers=4)
        #
        # for batch in loader:
        #     # 'batch["audio"]' will be shape [32, window_size_samples]
        #     pass