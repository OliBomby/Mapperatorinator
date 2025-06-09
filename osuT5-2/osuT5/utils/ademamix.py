import math
from typing import Iterable, Literal, Optional, Tuple

import torch

import bitsandbytes.functional as F
from bitsandbytes.optim.optimizer import Optimizer2State




class AdEMAMix(Optimizer2State):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        optim_bits: Literal[8, 32] = 32,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            "ademamix",
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optim_bits=optim_bits,
            args=None,
            min_8bit_size=min_8bit_size,
            percentile_clipping=100,
            block_wise=True,
            is_paged=is_paged,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
        )

    @torch.no_grad()
    def init_state(self, group, p, gindex, pindex):
        # In our AdEMAMix implementation, we use `state` to hold
        # both the fast and slow EMAs. Here we override the base
        # `Optimizer2State` to allocate a buffer twice as large.
        # Additional consideration: we do not support block_wise=False,
        # percentile clipping, or max_unorm.

        config = self.get_config(gindex, pindex, group)

        if config["optim_bits"] == 32:
            dtype = torch.float32
        elif config["optim_bits"] == 8:
            dtype = torch.uint8
        else:
            raise NotImplementedError(f'Amount of optimizer bits not supported: {config["optim_bits"]}')

        if p.numel() < config["min_8bit_size"]:
            dtype = torch.float32

        state = self.state[p]
        state["step"] = 0

        if dtype == torch.uint8:
            if "dynamic" not in self.name2qmap:
                self.fill_qmap()
            self.name2qmap["dynamic"] = state["qmap1"] = self.name2qmap["dynamic"].to(p.device)
            self.name2qmap["udynamic"] = state["qmap2"] = self.name2qmap["udynamic"].to(p.device)

            n = p.numel()
            blocks = (n // 256) + bool(n % 256)

            state["absmax1"] = torch.zeros((2, blocks), dtype=torch.float32, device=p.device)
            state["absmax2"] = torch.zeros((blocks,), dtype=torch.float32, device=p.device)

        state["state1"] = self._get_state_double_buffer(p, dtype=dtype)
        state["state2"] = self.get_state_buffer(p, dtype=dtype)

    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex):
        config = self.get_config(gindex, pindex, group)

        if config["t_alpha"] is None and config["t_beta3"] is None:
            # Not using alpha/beta3 scheduler; we can fall through.
            super().update_step(group, p, gindex, pindex)
            return

        # Ensure contiguous memory layout
        p.data = p.data.contiguous()
        p.grad = p.grad.contiguous()

        state = self.state[p]
        grad = p.grad

        state["step"] += 1
        step = state["step"]

        beta1, beta2, beta3 = config["betas"]
        alpha = config["alpha"]
        t_alpha = config["t_alpha"]
        t_beta3 = config["t_beta3"]

        # Apply scheduler for alpha
        if t_alpha is not None:
            alpha_t = min(step * alpha / t_alpha, alpha)
        else:
            alpha_t = alpha
            
        

        # Apply scheduler for beta3
        if t_beta3 is not None:
            ln_beta1 = math.log(beta1)
            ln_beta3 = math.log(beta3)
            step_scale = step / t_beta3
            beta3_t = min(
                math.exp((ln_beta1 * ln_beta3) / (((1 - step_scale) * ln_beta3) + (step_scale * ln_beta1))), beta3
            )
        else:
            beta3_t = beta3
            
        #print(f"apha_t: {alpha_t} beta3_t: {beta3_t}")

        # Apply updates
        if state["state1"].dtype == torch.float32:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                beta1,
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                beta2,
                beta3_t,
                alpha_t,
                config["weight_decay"],
                gnorm_scale=1.0,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )
        elif state["state1"].dtype == torch.uint8:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                beta3_t,
                alpha_t,
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=1.0,
                skip_zeros=config["skip_zeros"],
            )

    def _get_state_double_buffer(self, p, dtype=torch.float32):
        if not self.is_paged or p.numel() < 1e5:
            return torch.zeros((2, *p.size()), dtype=dtype, device=p.device)
        else:
            buff = F.get_paged(*(2, *p.size()), dtype=dtype, device=p.device)
            F.fill(buff, 0)
            self.page_mng.paged_tensors.append(buff)
            return buff


class AdEMAMix8bit(AdEMAMix):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
            eps=eps,
            weight_decay=weight_decay,
            optim_bits=8,
            min_8bit_size=min_8bit_size,
            is_paged=is_paged,
        )


class PagedAdEMAMix8bit(AdEMAMix8bit):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
            eps=eps,
            weight_decay=weight_decay,
            min_8bit_size=min_8bit_size,
            is_paged=True,
        )


class PagedAdEMAMix(AdEMAMix):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        optim_bits: Literal[8, 32] = 32,
        min_8bit_size: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
            eps=eps,
            weight_decay=weight_decay,
            optim_bits=optim_bits,
            min_8bit_size=min_8bit_size,
            is_paged=True,
        )


class AdEMAMix32bit(Optimizer2State):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
        is_paged: bool = False,
    ):
        super().__init__(
            "ademamix",
            params=params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            optim_bits=32,
            args=None,
            min_8bit_size=min_8bit_size,
            percentile_clipping=100,
            block_wise=True,
            is_paged=is_paged,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
        )


class PagedAdEMAMix32bit(AdEMAMix32bit):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: Tuple[float, float, float] = (0.9, 0.999, 0.9999),
        alpha: float = 5.0,
        t_alpha: Optional[int] = None,
        t_beta3: Optional[int] = None,
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
        min_8bit_size: int = 4096,
    ):
        super().__init__(
            params,
            lr=lr,
            betas=betas,
            alpha=alpha,
            t_alpha=t_alpha,
            t_beta3=t_beta3,
            eps=eps,
            weight_decay=weight_decay,
            min_8bit_size=min_8bit_size,
            is_paged=True,
        )