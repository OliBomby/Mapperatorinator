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
        accelerator.save_state(output_dir=output_dir, safe_serialization=False)
        


        """
        try:
            api.upload_folder(
            folder_path="../",
            path_in_repo="/backups",
            repo_id="Tiger14n/osuT5_backup",
            repo_type="model",
        )
        except Exception as e:
            print(e)
        """



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

def monitor_embedder_stats(model: nn.Module, batch: dict[str, torch.Tensor]):
    """
    Calculates and returns statistics (mean, std, max_abs) for the
    difficulty, global position, and mapper style embeddings for a given batch.

    Args:
        model: The OsuT model instance containing the embedders.
        batch: A dictionary containing the input data, expecting keys like
               'difficulty', 'global_pos', and 'mapper_idx'.

    Returns:
        A dictionary containing the calculated statistics for each embedding type.
        Returns NaN for stats if the corresponding embedder or input is missing.
    """
    stats = {}
    if not hasattr(model, 'difficulty_embedder') or \
       not hasattr(model, 'song_pos_embedder') or \
       not hasattr(model, 'mapper_embedder'):
        print("Warning: Model missing expected difficulty or global_pos embedder.")
        return stats # Cannot proceed without basic embedders

    # Use a representative device from the model
    device = next(model.parameters()).device

    with torch.no_grad(): # Disable gradient calculation for monitoring

        # --- 1. Difficulty Embedding Stats ---
        if 'difficulty' in batch and batch['difficulty'] is not None:
            try:
                difficulty_input = batch['difficulty'].to(device)

                # Ensure input shape is [B, 1] for the Linear layer
                if difficulty_input.dim() == 1:
                    difficulty_input = difficulty_input.unsqueeze(-1) # [B] -> [B, 1]
                elif difficulty_input.dim() == 2 and difficulty_input.shape[1] > 1:
                    # Assuming the first column is the intended difficulty scalar
                    print(f"Warning: Selecting first feature from difficulty tensor with shape {difficulty_input.shape} in monitor_embedder_stats.")
                    difficulty_input = difficulty_input[:, 0:1]
                elif difficulty_input.dim() != 2 or difficulty_input.shape[1] != 1:
                     raise ValueError(f"Unexpected shape for difficulty: {difficulty_input.shape}. Expected [B] or [B, 1].")

                # Normalize *after* ensuring correct shape
                norm_diff = difficulty_input / 10.0
                diff_emb = model.difficulty_embedder(norm_diff)

                stats.update({
                    'diff_emb_mean': diff_emb.mean().item(),
                    'diff_emb_std': diff_emb.std().item(),
                    'diff_emb_max_abs': diff_emb.abs().max().item(),
                })
            except Exception as e:
                print(f"Error processing difficulty embedding stats: {e}")
                stats.update({'diff_emb_mean': float('nan'), 'diff_emb_std': float('nan'), 'diff_emb_max_abs': float('nan')})
        else:
            stats.update({'diff_emb_mean': float('nan'), 'diff_emb_std': float('nan'), 'diff_emb_max_abs': float('nan')})


        # --- 2. Global Position Embedding Stats ---
        if 'global_pos' in batch and batch['global_pos'] is not None:
            try:
                global_pos_input = batch['global_pos'].to(device)

                # Ensure input shape is [B, 2]
                if global_pos_input.dim() != 2 or global_pos_input.shape[1] != 2:
                    raise ValueError(f"Unexpected shape for global_pos: {global_pos_input.shape}. Expected [B, 2].")

                pos_emb = model.song_pos_embedder(global_pos_input)

                stats.update({
                    'pos_emb_mean': pos_emb.mean().item(),
                    'pos_emb_std': pos_emb.std().item(),
                    'pos_emb_max_abs': pos_emb.abs().max().item(),
                })
            except Exception as e:
                print(f"Error processing global position embedding stats: {e}")
                stats.update({'pos_emb_mean': float('nan'), 'pos_emb_std': float('nan'), 'pos_emb_max_abs': float('nan')})
        else:
             stats.update({'pos_emb_mean': float('nan'), 'pos_emb_std': float('nan'), 'pos_emb_max_abs': float('nan')})


        # --- 3. Mapper Style Embedding Stats ---
        if hasattr(model, 'mapper_embedder') and 'mapper_idx' in batch and batch['mapper_idx'] is not None:
            try:
                mapper_ids_input = batch['mapper_idx'].to(device)

                # Ensure input shape is [B]
                if mapper_ids_input.dim() != 1:
                     raise ValueError(f"Unexpected shape for mapper_idx: {mapper_ids_input.shape}. Expected [B].")

                # Pass through the mapper embedder
                # Note: MapperStyleEmbedder handles the -1 mapping internally
                style_emb = model.mapper_embedder(mapper_ids_input)

                if style_emb is not None:
                     # Check if the embedder actually produced output (it handles None input)
                    stats.update({
                        'style_emb_mean': style_emb.mean().item(),
                        'style_emb_std': style_emb.std().item(),
                        'style_emb_max_abs': style_emb.abs().max().item(),
                        # Optional: Monitor the norm of the underlying embedding weights themselves (expensive if done often)
                        # 'style_emb_weight_norm': model.mapper_embedder.embedding.weight.norm().item()
                    })
                else:
                    # Embedder returned None
                    stats.update({'style_emb_mean': float('nan'), 'style_emb_std': float('nan'), 'style_emb_max_abs': float('nan')})

            except Exception as e:
                print(f"Error processing mapper style embedding stats: {e}")
                stats.update({'style_emb_mean': float('nan'), 'style_emb_std': float('nan'), 'style_emb_max_abs': float('nan')})
        else:
            # Embedder doesn't exist or mapper_idx not in batch
             stats.update({'style_emb_mean': float('nan'), 'style_emb_std': float('nan'), 'style_emb_max_abs': float('nan')})

    return stats


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
                                    tokenizer.event_end[EventType.VOLUME] + tokenizer.vocab_size_out),
            "total_acc": acc_range(preds, labels, 0, tokenizer.vocab_size_out),
                                    }
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

    import traceback
        
    if args.compile:
       model = torch.compile(model)
       
    model.train()
    train_averager = Averager()
    monitor_every = 100
    while shared.current_train_step <= args.optim.total_steps:
        optimizer.zero_grad(set_to_none=True)
        accelerator.print(f"Epoch {shared.current_epoch}")

        try:
            for batch_id, batch in enumerate(train_dataloader, start=1):
                try:
                    with accelerator.accumulate(model):
                        if shared.current_train_step > args.optim.total_steps:
                            break
                        
                        if shared.current_train_step % monitor_every == 0:
                            with torch.no_grad():  # Important to not interfere with training
                                 stats = monitor_embedder_stats(model, batch)
                                 embed_stats = {
                                        'emb_stats': stats,
                                    }
                                 accelerator.log(embed_stats, step=shared.current_train_step)
                                 

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
                          
                            maybe_eval(model, accelerator, test_dataloader, tokenizer, args, shared)

                            maybe_save_checkpoint(accelerator, args, shared, model)
                            if args.optim.lr_scheduler == "ReduceLROnPlateau":
                                # every eval step, we update the lr
                                if (shared.current_train_step % args.eval.every_steps == 0):
                                    lr_scheduler.step(metrics=shared.current_loss)
                            else:
                                lr_scheduler.step()

                            shared.current_train_step += 1
                            

                except Exception as e:
                   # accelerator.print(f"Error in batch {batch_id}:")
                 #  accelerator.print(f"Batch contents: {batch}")
                   # accelerator.print(f"Error details: {str(e)}")
                    raise  # Re-raise the exception to be caught by the outer try-except
            shared.current_epoch += 1
            
        except Exception as e:
            traceback.print_exc()
            accelerator.print(f"Error during training: {str(e)}. Retry.")
   
            
            
           

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
