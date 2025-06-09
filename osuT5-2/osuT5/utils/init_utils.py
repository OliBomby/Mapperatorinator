import torch
import os

from accelerate.utils import set_seed
from omegaconf import open_dict, DictConfig


def check_args_and_env(args: DictConfig) -> None:
    assert args.optim.batch_size % args.optim.grad_acc == 0
    # Train log must happen before eval log
    assert args.eval.every_steps % args.logging.every_steps == 0

    if args.device == "gpu":
        assert torch.cuda.is_available(), "We use GPU to train/eval the model"


def opti_flags(args: DictConfig) -> None:
    # This lines reduce training step by 2.4x
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


def update_args_with_env_info(args: DictConfig) -> None:
    with open_dict(args):
        slurm_id = os.getenv("SLURM_JOB_ID")

        if slurm_id is not None:
            args.slurm_id = slurm_id
        else:
            args.slurm_id = "none"

        args.working_dir = os.getcwd()


def setup_args(args: DictConfig) -> None:
    check_args_and_env(args)
    update_args_with_env_info(args)
    opti_flags(args)

    if args.seed is not None:
        set_seed(args.seed)
