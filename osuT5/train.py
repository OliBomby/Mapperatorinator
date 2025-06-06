import hydra
import torch
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration

from osuT5.config import TrainConfig
from osuT5.utils import (
    setup_args,
    train,
    train_profiling,
    get_model,
    get_tokenizer,
    get_scheduler,
    get_optimizer,
    get_dataloaders,
    get_shared_training_state,
)


@hydra.main(config_path="../configs/osut5", config_name="train_v29", version_base="1.1")
def main(args: TrainConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(
        cpu=args.device == "cpu",
        mixed_precision=args.precision,
        gradient_accumulation_steps=args.optim.grad_acc,
        log_with=args.logging.log_with,
        project_config=ProjectConfiguration(
            project_dir="..", logging_dir="tensorboard_logs"
        ),
        kwargs_handlers=[ddp_kwargs],
    )
    accelerator.init_trackers(
        "osuT5",
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                "job_type": "training",
                "sync_tensorboard": args.profile.do_profile,
                "mode": args.logging.mode,
            }
        }
    )

    setup_args(args)

    shared = get_shared_training_state()
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, accelerator)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)

    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path)
        if args.pretrained_t5_compat:
            del state_dict["shared.weight"]
            del state_dict["encoder.embed_tokens.weight"]
            del state_dict["decoder.embed_tokens.weight"]
            del state_dict["lm_head.weight"]
            model.transformer.load_state_dict(state_dict, strict=False)
        else:
            model.load_state_dict(state_dict)

    if args.model.manual_norm_weights:
        print("Manually normalizing model weights")
        model.transformer.register_step_post_hook(optimizer)
        model.transformer.norm_weights_()

    # noinspection PyTypeChecker
    (
        model,
        optimizer,
        scheduler,
        train_dataloader,
        test_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, train_dataloader, test_dataloader
    )

    accelerator.register_for_checkpointing(tokenizer)

    if args.checkpoint_path:
        accelerator.load_state(args.checkpoint_path)
        shared.current_train_step = scheduler.scheduler.last_epoch // accelerator.num_processes + 1

    if args.compile:
        model = torch.compile(model)

    func = train_profiling if args.profile.do_profile else train

    func(
        model,
        train_dataloader,
        test_dataloader,
        accelerator,
        scheduler,
        optimizer,
        tokenizer,
        args,
        shared,
    )


if __name__ == "__main__":
    main()
