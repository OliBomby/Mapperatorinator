import glob
import os.path
import time
from multiprocessing.managers import Namespace
import huggingface_hub as api
import torch
import wandb
from accelerate import Accelerator
from accelerate.logging import get_logger
from omegaconf import DictConfig
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from typing import Callable

from ..dataset.ors_dataset import LABEL_IGNORE_ID
from ..tokenizer import Tokenizer, EventType, ContextType
from ..model import OsuT
from .log_utils import Averager

logger = get_logger(__name__)



def forward(model: OsuT, batch):
    outputs = model(**batch)
    loss = outputs.loss

    stats = {"loss": loss.detach()}
    return loss, stats


def forward_eval(model: OsuT, batch):
    outputs = model(**batch)
    return outputs


def add_prefix(prefix: str, stats: dict[str, float]):
    return {f"{prefix}/{k}": v for k, v in stats.items()}


def maybe_save_checkpoint(accelerator: Accelerator, args: DictConfig, shared: Namespace, model: OsuT):
    if (
            shared.current_train_step > args.optim.total_steps
            or shared.current_train_step % args.checkpoint.every_steps == 0
    ):
        accelerator.wait_for_everyone()

        if not accelerator.is_main_process:
            return

        if shared.current_loss < shared.best_loss:
            shared.best_loss = shared.current_loss
            is_best = True
        else:
            is_best = False

        output_dir = f"checkpoint-{shared.current_train_step}"
        # Saving T5 has an issue that safe serialization removes shared tensors and then the model can't be loaded.
        unwrapped_model = accelerator.unwrap_model(model)
        lora_params = {n: p for n, p in unwrapped_model.named_parameters() if "lora" in n}
        for n, p in lora_params.items():
            print(n, p.sum())
            
        unwrapped_model.save_pretrained(output_dir)
            



def maybe_eval(
        model: OsuT,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    if (
            shared.current_train_step > args.optim.total_steps
            or shared.current_train_step % args.eval.every_steps == 0
    ):
        model.eval()

        with torch.no_grad():
            eval_model(model, accelerator, dataloader, tokenizer, args, shared)

        shared.last_log = time.time()
        model.train()


def maybe_logging(
        model: OsuT,
        accelerator: Accelerator,
        optimizer: Optimizer,
        averager: Averager,
        args: DictConfig,
        shared: Namespace,
):
    def extra_stats(args, shared, model, optimizer):
        stats = {}

        if args.logging.weights_l2:
            weights_l2 = (
                    sum(p.detach().norm(2).item() ** 2 for p in model.parameters() if p.requires_grad) ** 0.5
            )
            stats["weights_l2"] = weights_l2

        stats["lr"] = optimizer.param_groups[0]["lr"]
        stats["seconds_per_step"] = (
                                            time.time() - shared.last_log
                                    ) / args.logging.every_steps

        return stats

    if shared.current_train_step % args.logging.every_steps == 0:
        stats = extra_stats(args, shared, model, optimizer)

        averager.update(stats)
        averaged_stats = averager.average()
        averaged_stats["epoch"] = shared.current_epoch
        averaged_stats = add_prefix("train", averaged_stats)
        accelerator.log(averaged_stats, step=shared.current_train_step)
        averaged_stats["step"] = shared.current_train_step
        logger.info(averaged_stats)

        shared.last_log = time.time()


def maybe_grad_clip_and_grad_calc(
        model: OsuT,
        accelerator: Accelerator,
        args: DictConfig,
):
    if args.optim.grad_clip > 0:
        grad_l2 = accelerator.clip_grad_norm_(
            parameters=model.parameters(),
            max_norm=args.optim.grad_clip,
            norm_type=2,
        ).item()
    else:
        grad_l2 = None

    if args.logging.grad_l2:
        if grad_l2 is None:
            grad_l2 = (
                    sum(
                        p.grad.detach().data.norm(2).item() ** 2 for p in model.parameters()
                    )
                    ** 0.5
            )

        return {"grad_l2": grad_l2}
    else:
        return {}


# noinspection PyUnresolvedReferences,PyTypeChecker
@torch.no_grad()
def eval_model(
        model: OsuT,
        accelerator: Accelerator,
        dataloader: DataLoader,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    shared.last_log = time.time()
    averager = Averager()

    class_weights = torch.ones(tokenizer.vocab_size_out)
    class_weights[tokenizer.event_start[EventType.TIME_SHIFT]:tokenizer.event_end[EventType.TIME_SHIFT]] = args.data.rhythm_weight
    loss_fn = nn.CrossEntropyLoss(weight=class_weights, reduction="none", ignore_index=LABEL_IGNORE_ID)
    loss_fn = loss_fn.to(accelerator.device)

    for batch_id, batch in enumerate(dataloader, start=1):
        if batch_id == args.eval.steps * args.optim.grad_acc:
            break

        # We can't use the beatmap idx of the test set because these are not known by the model
       # del batch["beatmap_idx"]

        outputs = forward_eval(model, batch)

        # Reduce loss over all processes
        loss = outputs.loss
        loss = accelerator.reduce(loss, reduction="mean")

        # Gether labels and predictions over all processes and drop duplicates
        logits = outputs.logits
        preds = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        accelerator.gather_for_metrics((logits, preds, labels))

        # Calculate accuracy metrics
        if len(args.data.context_types) > 0:
            for cts in args.data.context_types:
                ct = ContextType(cts)
                ct_index = batch['decoder_input_ids'][:, 0] == tokenizer.context_sos[ct]

                if not ct_index.any():
                    continue

                ct_logits = outputs.logits[ct_index]
                ct_preds = preds[ct_index]
                ct_labels = labels[ct_index]
                ct_weights = batch["sample_weights"][ct_index] if "sample_weights" in batch else None
                ct_loss = calc_loss(loss_fn, ct_logits, ct_labels, ct_weights)
                stats = get_stats(ct_loss, ct_preds, ct_labels, tokenizer, args)

                if ct != ContextType.NONE:
                    stats = add_prefix(cts, stats)

                averager.update(stats)
        else:
            stats = get_stats(loss, preds, labels, tokenizer, args)
            averager.update(stats)




    averager.update({"time": time.time() - shared.last_log})
    averaged_stats = averager.average()
    averaged_stats = add_prefix("test", averaged_stats)
    accelerator.log(averaged_stats, step=shared.current_train_step)
    logger.info(averaged_stats)

    if "test/loss" in averaged_stats:
        shared.current_loss = averaged_stats["test/loss"]


def calc_loss(loss_fn, logits, labels, sample_weights):
    unreduced_loss = loss_fn(torch.swapaxes(logits, 1, -1), labels)
    if sample_weights is not None:
        unreduced_loss *= sample_weights.unsqueeze(1)
    return unreduced_loss.sum() / (labels != LABEL_IGNORE_ID).sum()


def get_stats(loss, preds, labels, tokenizer, args: DictConfig):
    stats = {"loss": loss.detach(),
             "timing_acc": acc_range(preds, labels, tokenizer.event_start[EventType.TIME_SHIFT],
                                     tokenizer.event_end[EventType.TIME_SHIFT]),
             "hitsound_acc": acc_range(preds, labels, tokenizer.event_start[EventType.HITSOUND],
                                       tokenizer.event_end[EventType.HITSOUND]),
             "volume_acc": acc_range(preds, labels, tokenizer.event_start[EventType.VOLUME],
                                     tokenizer.event_end[EventType.VOLUME]),
             "other_acc": acc_range(preds, labels, tokenizer.event_end[EventType.VOLUME],
                                    tokenizer.event_end[EventType.VOLUME] + tokenizer.vocab_size_out)}
    if args.data.add_positions:
        if args.data.position_split_axes:
            stats["position_acc"] = acc_range(preds, labels, tokenizer.event_start[EventType.POS_X],
                                              tokenizer.event_end[EventType.POS_Y])
        else:
            stats["position_acc"] = acc_range(preds, labels, tokenizer.event_start[EventType.POS],
                                              tokenizer.event_end[EventType.POS])
    if args.data.add_distances:
        stats["spacing_acc"] = acc_range(preds, labels, tokenizer.event_start[EventType.DISTANCE],
                                         tokenizer.event_end[EventType.DISTANCE])
    return stats


def acc_range(preds, labels, start_index, end_index):
    index = (start_index <= labels) & (labels < end_index)
    range_labels = labels[index]
    range_preds = preds[index]
    accs = range_preds == range_labels
    if isinstance(accs, torch.Tensor):
        accs = accs.detach().float().cpu().numpy()
    return accs


def fuzzy_acc_range(preds, labels, start_index, end_index, fuzzyness=0):
    index = (start_index <= labels) & (labels < end_index)
    range_labels = labels[index]
    range_preds = preds[index]
    accs = (range_preds - fuzzyness <= range_labels) & (range_labels <= range_preds + fuzzyness)
    if isinstance(accs, torch.Tensor):
        accs = accs.detach().float().cpu().numpy()
    return accs


def train(
        model: OsuT,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        lr_scheduler: LRScheduler,
        optimizer: Optimizer,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
        profiler=None,
        max_retries: int = 20,
        
        
):
    
    uncompiled_model = model
    if args.compile:
        model = torch.compile(uncompiled_model)

    import traceback
    model.train()
    train_averager = Averager()
    tries = 0
    monitor_every = 100
    with torch.no_grad():
        eval_model(uncompiled_model, accelerator, test_dataloader, tokenizer, args, shared)

    while shared.current_train_step <= args.optim.total_steps:
        optimizer.zero_grad(set_to_none=True)
        accelerator.print(f"Epoch {shared.current_epoch}")

        try:
            for batch_id, batch in enumerate(train_dataloader, start=1):
                try:
                    with accelerator.accumulate(model):
                        if shared.current_train_step > args.optim.total_steps:
                            break


                        loss, stats = forward(model, batch)

                        accelerator.backward(loss)
                        train_averager.update(stats)

                        if accelerator.sync_gradients:
                            stats = maybe_grad_clip_and_grad_calc(model, accelerator, args)
                            train_averager.update(stats)

                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)

                        if profiler is not None:
                            profiler.step()

                        if accelerator.sync_gradients:
                            maybe_logging(model, accelerator, optimizer, train_averager, args, shared)
                            maybe_eval(uncompiled_model, accelerator, test_dataloader, tokenizer, args, shared)
                            maybe_save_checkpoint(accelerator, args, shared, model)
                            if args.optim.lr_scheduler == "ReduceLROnPlateau":
                                # every eval step, we update the lr
                                if (shared.current_train_step % args.eval.every_steps == 0):
                                    lr_scheduler.step(metrics=shared.current_loss)
                            else:
                                lr_scheduler.step()

                            shared.current_train_step += 1
                            
                        tries = 0

                except Exception as e:
                   # accelerator.print(f"Error in batch {batch_id}:")
                 #  accelerator.print(f"Batch contents: {batch}")
                   # accelerator.print(f"Error details: {str(e)}")
                    raise  # Re-raise the exception to be caught by the outer try-except
            shared.current_epoch += 1
            
        except Exception as e:
            traceback.print_exc()
            tries += 1
            accelerator.print(f"Error during training: {str(e)}. Retry.")
            if tries >= max_retries:
                raise e
            
            
           

    if not (args.profile.do_profile and args.profile.early_stop):
        maybe_eval(model, accelerator, test_dataloader, tokenizer, args, shared)
        maybe_save_checkpoint(accelerator, args, shared, model)

    accelerator.end_training()


def train_profiling(
        model: OsuT,
        train_dataloader: DataLoader,
        test_dataloader: DataLoader,
        accelerator: Accelerator,
        lr_scheduler: LRScheduler,
        optimizer: Optimizer,
        tokenizer: Tokenizer,
        args: DictConfig,
        shared: Namespace,
):
    tensorboard_trace_handler = torch.profiler.tensorboard_trace_handler(
        "./profiler_logs", worker_name=f"worker_{accelerator.process_index}")

    if args.profile.early_stop:
        stop_step = (args.profile.wait + args.profile.warmup + args.profile.active) * args.profile.repeat / args.optim.grad_acc
        args.optim.total_steps = shared.current_train_step + stop_step

    def on_trace_ready(trace):
        tensorboard_trace_handler(trace)
        wandb_tracker = accelerator.get_tracker("wandb")
        if wandb_tracker is not None:
            wandb.save(glob.glob(f"./profiler_logs/*.pt.trace.json")[0], base_path="profiler_logs")

    with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=args.profile.wait,
                warmup=args.profile.warmup,
                active=args.profile.active,
                repeat=args.profile.repeat,
            ),
            on_trace_ready=on_trace_ready,
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
    ) as p:
        train(
            model,
            train_dataloader,
            test_dataloader,
            accelerator,
            lr_scheduler,
            optimizer,
            tokenizer,
            args,
            shared,
            p
        )
