import os
TORCH_CUDNN_SDPA_ENABLED=1
import hydra
import torch
import multiprocessing
import time
torch._dynamo.config.cache_size_limit = 64
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import ProjectConfiguration
from omegaconf import DictConfig
import os
import glob
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

def make_inputs_require_grad(module, input, output):
    output.requires_grad_(True)



def print_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())  # Total parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)  # Trainable params
    frozen_params = total_params - trainable_params  # Non-trainable (frozen) params

    print(f"Total Parameters: {total_params:,}")
    print(f"Trainable Parameters: {trainable_params:,}")
    print(f"Frozen Parameters: {frozen_params:,}")

@hydra.main(config_path="./configs/osuT5", config_name="train_v23", version_base="1.1")
def main(args: DictConfig):
    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
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
    run_name = f"{args.model.name}_{args.optim.name}_{time.strftime('%Y-%m-%d_%H-%M-%S')}"
    accelerator.init_trackers(
        "osuT5",  # Project name
        init_kwargs={
            "wandb": {
                "entity": "mappingtools",
                'name': run_name,
                "job_type": "training",
                "config": dict(args),
                "sync_tensorboard": args.profile.do_profile,
                "mode": args.logging.mode,
            }
        }
    )

    setup_args(args)
    torch.cuda.empty_cache()
    shared = get_shared_training_state()
    tokenizer = get_tokenizer(args)
    model = get_model(args, tokenizer)
    train_dataloader, test_dataloader = get_dataloaders(tokenizer, args, shared)



    if args.pretrained_path:
        state_dict = torch.load(args.pretrained_path)
        if args.pretrained_t5_compat:
            del state_dict["shared.weight"]
            del state_dict["encoder.embed_tokens.weight"]
            del state_dict["decoder.embed_tokens.weight"]
            del state_dict["lm_head.weight"]
            model.transformer.load_state_dict(state_dict, strict=False)
        elif args.lora:
            model.load_state_dict(state_dict)
            model.to(accelerator.device)

            # 2. Access the original (non-optimized) model

            from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
            config = LoraConfig(r=64, lora_alpha=128,
                        target_modules=['k_proj', 'v_proj', 'q_proj', 'out_proj'], lora_dropout=0.05, bias="none")
            model = get_peft_model(model, config)
            model.model.transformer.model.encoder.conv1.register_forward_hook(make_inputs_require_grad)
            lora_params = {n: p for n, p in model.named_parameters() if "lora" in n}
            for n, p in lora_params.items():
                print(n, p.sum())
            accelerator.print(model.print_trainable_parameters())
        elif args.prune:
 
            model.load_state_dict(state_dict)

            # freeze encoder
            model.transformer.model.freeze_encoder()

            # there are 12 decoder layers, remove all decoder layers accept 1, 4, 8 ,and 12
            model.transformer.model.decoder.layers = torch.nn.ModuleList([model.transformer.model.decoder.layers[i] for i in [0, 11]])


        else:

            model.load_state_dict(state_dict)
            model.to(accelerator.device)

    
    print_model_parameters(model)       
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args, accelerator)

    
    #model.transformer.gradient_checkpointing_enable()
    (
        model,
        optimizer,
        scheduler,
        test_dataloader,
        train_dataloader,
    ) = accelerator.prepare(
        model, optimizer, scheduler, test_dataloader, train_dataloader
    )

    accelerator.register_for_checkpointing(tokenizer)

    
        # Example usage in loading the checkpoint
    if args.checkpoint_path and not args.auto_checkpoint:
        accelerator.load_state(args.checkpoint_path)
        shared.current_train_step = scheduler.scheduler.last_epoch // accelerator.num_processes + 1



    func = train_profiling if args.profile.do_profile else train
    
    func(
        model,
        train_dataloader,  # Pass the callable
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
