import string
import random
import numpy as np
import torch
import math
from torch.distributed.tensor import DTensor, distribute_tensor

from .utils import to_dist, to_local

# torch._dynamo.config.cache_size_limit = 1_000_000

# try:
#     torch.backends.opt_einsum.strategy = "dynamic-programming"
# except AttributeError:
#     # opt_einsum backend is not available, so we'll skip setting the strategy
#     pass

# adapted from https://github.com/evanatyourservice/kron_torch/blob/mars/kron_torch/kron.py

def is_tensor(x):
    return isinstance(x, torch.Tensor)


def get_q(state):
    return [q for q in (state["Q0"], state["Q1"], state["Q2"], state["Q3"]) if is_tensor(q)]


class KronMars(torch.optim.Optimizer):
    """Implements PSGD Kron from https://github.com/lixilinx/psgd_torch.

    Args:
        params (iterable): Iterable of parameters to optimize or dicts defining
            parameter groups.
        lr (float): Learning rate.
        momentum (float): Momentum parameter.
        mars_beta (float): MARS beta parameter.
        weight_decay (float): Weight decay (L2 penalty).
        preconditioner_update_probability (callable or float, optional): Probability of
            updating the preconditioner. If None, defaults to a schedule that anneals
            from 1.0 to 0.03 by 4000 steps.
        max_size_triangular (int): Max size for dim's preconditioner to be triangular.
        min_ndim_triangular (int): Minimum number of dimensions a layer needs
            to have triangular preconditioners.
        memory_save_mode: (string, optional), None, 'one_diag', or 'all_diag', None is default
            to set all preconditioners to be triangular, 'one_diag' sets the largest
            or last dim to be diagonal per layer, and 'all_diag' sets all preconditioners
            to be diagonal.
        momentum_into_precond_update: (bool), whether to send momentum into preconditioner
            update instead of raw gradients.
        mu_dtype (torch.dtype, optional): Dtype of the momentum accumulator.
        precond_dtype (torch.dtype, optional): Dtype of the preconditioner.
        verbose (bool): Whether to print energy statistics.
        use_grad_stats (bool): Whether to use gradient statistics for preconditioner update.
        std_scale (float): Scale factor for the standard deviation in the fake momentum calculation.
        precond_lr (float): Learning rate for preconditioner update.
        precond_lr_schedule (callable or None): Schedule for preconditioner learning rate.
        gamma (float): MARS gamma parameter.
    """

    def __init__(
        self,
        params,
        lr=0.001,
        momentum=0.9,
        mars_beta=0.9,
        weight_decay=0.0,
        preconditioner_update_probability_schedule=True,
        max_size_triangular=8192,
        min_ndim_triangular=2,
        memory_save_mode=None,
        momentum_into_precond_update=True,
        mu_dtype=None,
        precond_dtype=None,
        verbose=False,
        use_grad_stats=True,
        std_scale=1.0,
        precond_lr=0.1,
        precond_lr_schedule=None,
        gamma=0.05,
        # schedule
        min_prob=0.03,
        max_prob=1.0,
        decay=0.001,
        flat_start=250,
    ):

        defaults = dict(
            lr=lr,
            momentum=momentum,
            mars_beta=mars_beta,
            weight_decay=weight_decay,
            preconditioner_update_probability_schedule=preconditioner_update_probability_schedule,
            max_size_triangular=max_size_triangular,
            min_ndim_triangular=min_ndim_triangular,
            memory_save_mode=memory_save_mode,
            momentum_into_precond_update=momentum_into_precond_update,
            precond_lr=precond_lr,
            precond_lr_schedule=precond_lr_schedule,
            precond_init_scale=2.0,
            mu_dtype=mu_dtype,
            precond_dtype=precond_dtype,
            verbose=verbose,
            use_grad_stats=use_grad_stats,
            std_scale=std_scale,
            gamma=gamma,
            # pickling for statedict is picky with what it will allow, only int, float, and tensor, no functions can be in state
            max_prob=max_prob,
            min_prob=min_prob,
            decay=decay,
            flat_start=flat_start,

        )
        super(KronMars, self).__init__(params, defaults)

        self._tiny = torch.finfo(torch.bfloat16).tiny
        self._eps = torch.finfo(torch.bfloat16).eps
        self._prob_step = 0
        self.rng = random.Random(5318008)
        self.momentum_mean = {}
        self.momentum_var = {}
        self.momentum_count = {}
        self.momentum_energies = []
        self.pre_grad_energies = []
        self.fake_momentum_energies = []
        self.expr_cache = {}

    def precond_update_prob_schedule(self, n, max_prob=1.0, min_prob=0.03, decay=0.001, flat_start=250):
        """Anneal preconditioner update probability during beginning of training.

        PSGD benefits from more preconditioner updates at the beginning of training,
        but once the preconditioner is learned the update probability can drop low.

        This schedule is an exponential anneal with a flat start. Default settings keep
        update probability at 1.0 for 200 steps then exponentially anneal down to
        `min_prob` by 4000 steps. Default settings work very well for most models and
        training regimes.
        """

        """Exponential anneal with flat start."""
        n = torch.tensor(n, dtype=torch.float32)
        prob = max_prob * torch.exp(-decay * (n - flat_start))
        prob.clamp_(min=min_prob, max=max_prob)
        return prob


    def init_state(self, state, param, group, **kwargs):
        mu_dtype = kwargs.get("mu_dtype")
        precond_dtype = kwargs.get("precond_dtype")

        total_momentum_size = 0
        total_momentum_mb = 0
        total_precond_size = 0
        total_precond_mb = 0

        state["step"] = 0
        state["momentum_buffer"] = torch.zeros_like(param, dtype=mu_dtype or param.dtype)
        state["prev_grad"] = torch.zeros_like(param)
        q_state, exprs = init_Q_exprs(
            param,
            group["precond_init_scale"],
            group["max_size_triangular"],
            group["min_ndim_triangular"],
            group["memory_save_mode"],
            dtype=precond_dtype,
        )
        state["pre_grad_energy"] = 0.0
        state["Q0"] = q_state[0]
        state["Q1"] = q_state[1]
        state["Q2"] = q_state[2]
        state["Q3"] = q_state[3]

        self.expr_cache[param._cdata] = exprs

        # Print sizes
        momentum_size = state["momentum_buffer"].numel()
        momentum_mb = momentum_size * state["momentum_buffer"].element_size() / (2**20)
        total_momentum_size += momentum_size
        total_momentum_mb += momentum_mb

        precond_size = sum(q.numel() for q in get_q(state))
        precond_mb = sum(q.numel() * q.element_size() for q in get_q(state)) / (2**20)
        total_precond_size += precond_size
        total_precond_mb += precond_mb

        # Initialize statistics
        self.momentum_mean[param] = torch.zeros_like(param)
        self.momentum_var[param] = torch.zeros_like(param)
        self.momentum_count[param] = 0

        if total_momentum_size > 0:
            print(f"PSGD Momentum buffer size: {total_momentum_size} " f"elements, {total_momentum_mb:.2f} MB")
            print(f"PSGD Preconditioners size: {total_precond_size} " f"elements, {total_precond_mb:.2f} MB")
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        momentum_energies = []
        fake_momentum_energies = []
        pre_grad_energies = []

        # update preconditioners all together
        if self.param_groups[0]["preconditioner_update_probability_schedule"] is True:
            update_prob = self.precond_update_prob_schedule(
                self._prob_step,
                self.param_groups[0]["max_prob"],
                self.param_groups[0]["min_prob"],
                self.param_groups[0]["decay"],
                self.param_groups[0]["flat_start"],
            )
        else:
            raise ValueError("Only True is supported for preconditioner_update_probability_schedule")
        do_update = self.rng.random() < update_prob
        self._prob_step += 1

        balance = self.rng.random() < 0.01 and do_update

        for group in self.param_groups:
            precond_dtype = group.get("precond_dtype", torch.float32)
            mu_dtype = group.get("mu_dtype")

            # Get current precond_lr based on schedule
            precond_lr = group['precond_lr']
            if group['precond_lr_schedule'] is not None:
                precond_lr = group['precond_lr_schedule'](self._prob_step)

            for i, p in enumerate(group["params"]):
                grad = p.grad
                if grad is None:
                    continue

                grad_norm = (1e-16 + grad.square().sum().sqrt())
                if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                    print(f"Warning: grad_norm is {grad_norm}, skipping update")
                    continue
                grad = grad / grad_norm
                state = self.state[p]

                if len(state) == 0:
                    self.init_state(state, p, group, mu_dtype=mu_dtype, precond_dtype=precond_dtype)

                state["step"] += 1

                # Calculate MARS correction term (without clipping)
                prev_grad = state["prev_grad"]
                correction = group["gamma"] * group["mars_beta"] / (1 - group["mars_beta"]) * (grad - prev_grad)
                c_t = grad + correction

                # Store current gradient for next iteration
                state["prev_grad"].copy_(grad)

                # Update momentum buffer with corrected gradient
                momentum_buffer = state["momentum_buffer"]
                momentum_buffer.lerp_(c_t, weight=1 - group["momentum"])
                
                # Apply Nesterov update with corrected gradient
                nesterov_momentum = momentum_buffer.lerp(c_t, 1 - group["momentum"])
                # Restore momentum dtype
                if mu_dtype is not None:
                    nesterov_momentum.copy_(
                        nesterov_momentum.to(dtype=mu_dtype, non_blocking=True)
                    )
                nesterov_momentum = nesterov_momentum.to(
                    dtype=precond_dtype, non_blocking=True
                )

                # this time around, initializing the randn tensor and distributing it will not bode well, we would need to broadcast first
                # but this op is full of operations that fail with dtensor anyway and in-place bits so we'll do it all local anyway
                # we're likely better off just bringing everything local than just sharding back at the end
                # cant do max either
                meta = None
                if isinstance(state["Q0"], DTensor):
                    for i in range(4):
                        if is_tensor(state[f"Q{i}"]):
                            q_state, meta = to_local(state[f"Q{i}"], keep_sharded=False)
                            state[f"Q{i}"] = q_state

                # balance preconditioners about every 100 updates
                if grad.dim() > 1 and balance:
                    _balance_Q(get_q(state))

                # Update preconditioner
                gathered_update_grad = False
                if do_update:
                    if group["use_grad_stats"] and self.momentum_count[p] > 0:
                        mean = self.momentum_mean[p]
                        # Add clipping to prevent extreme values
                        var = torch.clamp(self.momentum_var[p], min=0, max=1e6)
                        std = torch.sqrt(var / self.momentum_count[p] + self._eps)
                        
                        # Check for nan/inf before creating fake momentum
                        if torch.isnan(std).any() or torch.isinf(std).any():
                            print("Warning: std contains nan/inf, using momentum buffer instead")
                            update_grad = nesterov_momentum if group["momentum_into_precond_update"] else grad
                        else:
                            noise = torch.randn_like(momentum_buffer, dtype=precond_dtype)
                            # Clip the noise to prevent extreme values
                            noise = torch.clamp(noise, min=-3, max=3)
                            fake_momentum = mean + group["std_scale"] * std * noise
                            
                            if group["verbose"]:
                                fake_momentum_energy = torch.mean(fake_momentum**2).item()
                                if not (math.isnan(fake_momentum_energy) or math.isinf(fake_momentum_energy)):
                                    fake_momentum_energies.append(fake_momentum_energy)
                            
                            update_grad = fake_momentum if group["momentum_into_precond_update"] else grad
                        
                        # Reset statistics
                        self.momentum_mean[p].zero_()
                        self.momentum_var[p].zero_()
                        self.momentum_count[p] = 0
                    else:
                        update_grad = nesterov_momentum if group["momentum_into_precond_update"] else grad

                    if group["verbose"]:
                        momentum_energy = torch.mean(momentum_buffer**2).item()
                        if not (math.isnan(momentum_energy) or math.isinf(momentum_energy)):
                            momentum_energies.append(momentum_energy)

                    # Check update_grad for nan/inf before updating preconditioner
                    if torch.isnan(update_grad).any() or torch.isinf(update_grad).any():
                        print("Warning: update_grad contains nan/inf, skipping preconditioner update")
                        continue

                    if isinstance(update_grad, DTensor):
                        update_grad = to_local(update_grad, keep_sharded=False)[0]
                    gathered_update_grad = True
                    
                    _update_precond(
                        get_q(state),
                        self.expr_cache[p._cdata],
                        torch.randn(momentum_buffer.shape, device=momentum_buffer.device, dtype=precond_dtype),
                        update_grad,
                        precond_lr,
                        self._tiny,
                    )

                if not gathered_update_grad:
                    if isinstance(nesterov_momentum, DTensor):
                        update_grad = to_local(update_grad, keep_sharded=False)[0]


                pre_grad = _precond_grad(
                    get_q(state),
                    self.expr_cache[p._cdata],
                    update_grad,
                ).to(dtype=p.dtype, non_blocking=True)

                # Store pre_grad_energy in state
                state["pre_grad_energy"] = torch.mean(pre_grad**2).item()

                if group["verbose"]:
                    pre_grad_energy = state["pre_grad_energy"]
                    if not (math.isnan(pre_grad_energy) or math.isinf(pre_grad_energy)):
                        pre_grad_energies.append(pre_grad_energy)

                # now we can distribute again
                # meta will be None if we didn't need to distribute
                if meta is not None:
                    pre_grad = to_dist(
                        pre_grad,
                        **meta,
                    )

                    for i in range(4):
                        if is_tensor(state[f"Q{i}"]):
                            state[f"Q{i}"] = to_dist(
                                state[f"Q{i}"],
                                **meta,
                            )


                    state["momentum_buffer"] = to_dist(
                        momentum_buffer,
                        **meta,
                    )
                else:
                    state["momentum_buffer"].copy_(momentum_buffer)

                # Apply weight decay and update parameters
                if group["weight_decay"] != 0 and p.dim() >= 2:
                    pre_grad.add_(p, alpha=group["weight_decay"])
                p.add_(pre_grad, alpha=-group["lr"])

                # Update momentum statistics when not updating preconditioner
                if not do_update and group["use_grad_stats"]:
                    count = self.momentum_count[p]
                    new_count = count + 1
                    delta = momentum_buffer - self.momentum_mean[p]
                    self.momentum_mean[p] += delta / new_count
                    delta2 = momentum_buffer - self.momentum_mean[p]
                    self.momentum_var[p] += delta * delta2
                    self.momentum_count[p] += 1

        # Print energies if verbose
        if any(group["verbose"] for group in self.param_groups):
            if momentum_energies:
                mean_momentum_energy = sum(momentum_energies) / len(momentum_energies)
                print(f"Mean momentum buffer energy: {mean_momentum_energy:.6f}")
            
            if pre_grad_energies:
                mean_pre_grad_energy = sum(pre_grad_energies) / len(pre_grad_energies)
                print(f"Mean preconditioned gradient energy: {mean_pre_grad_energy:.6f}")
            
            if fake_momentum_energies:
                mean_fake_momentum_energy = sum(fake_momentum_energies) / len(fake_momentum_energies)
                print(f"Mean fake momentum energy: {mean_fake_momentum_energy:.6f}")

        # Clear energy lists at the start of each step
        self.momentum_energies.clear()
        self.pre_grad_energies.clear()
        self.fake_momentum_energies.clear()

        # Print energies if verbose
        if any(group["verbose"] for group in self.param_groups):
            if momentum_energies:
                self.momentum_energies.extend(momentum_energies)
            if pre_grad_energies:
                self.pre_grad_energies.extend(pre_grad_energies)
            if fake_momentum_energies:
                self.fake_momentum_energies.extend(fake_momentum_energies)

        return loss


def init_Q_exprs(t, scale, max_size, min_ndim_triangular, memory_save_mode, dtype=None):
    """For a scalar or tensor t, we initialize its preconditioner Q and
    reusable einsum expressions for updating Q and preconditioning gradient.
    """
    letters = string.ascii_lowercase + string.ascii_uppercase

    dtype = dtype if dtype is not None else t.dtype
    shape = t.shape
    if len(shape) == 0:  # scalar
        if isinstance(t, DTensor):
            Q = [
                distribute_tensor(
                    scale * torch.ones_like(t.full_tensor(), dtype=dtype),
                    device_mesh=t.device_mesh,
                    placements=t.placements,
                )
            ]
        else:
            Q = [scale * torch.ones_like(t, dtype=dtype)]
        for i in range(3):
            Q.append(-2)
        exprA = ",->"
        exprGs = [",->"]
        exprP = ",,->"
    else:  # tensor
        if len(shape) > 13:
            raise ValueError(f"Got tensor with dim {len(t.shape)}; Einstein runs out of letters!")

        scale = scale ** (1 / len(shape))

        if memory_save_mode is None:
            dim_diag = [False for _ in shape]
        elif memory_save_mode == "one_diag":
            rev_sorted_dims = np.argsort(shape)[::-1]
            dim_diag = [False for _ in shape]
            dim_diag[rev_sorted_dims[0]] = True
        elif memory_save_mode == "all_diag":
            dim_diag = [True for _ in shape]
        else:
            raise ValueError(
                f"Invalid memory_save_mode: {memory_save_mode}, must be one of " "[None, 'one_diag', 'all_diag']"
            )

        Q = []
        piece1A, piece2A, piece3A = ([], "", "")
        exprGs = []
        piece1P, piece2P, piece3P, piece4P = ([], [], "", "")
        for i, (size, dim_d) in enumerate(zip(shape, dim_diag)):
            if size == 1 or size > max_size or len(shape) < min_ndim_triangular or dim_d:
                # use diagonal matrix as preconditioner for this dim
                tensor = scale * torch.ones(size, dtype=dtype, device=t.device)
                if isinstance(t, DTensor):
                    # again special case where we dont need to broadcast because init matrix is same on all devices
                    tensor = distribute_tensor(tensor, device_mesh=t.device_mesh, placements=t.placements)
                Q.append(tensor)

                piece1A.append(letters[i])
                piece2A = piece2A + letters[i]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
                subscripts = piece1 + "," + piece1 + "->" + letters[i + 13]
                exprGs.append(subscripts)

                piece1P.append(letters[i + 13])
                piece2P.append(letters[i + 13])
                piece3P = piece3P + letters[i + 13]
                piece4P = piece4P + letters[i + 13]
            else:
                # use triangular matrix as preconditioner for this dim
                tensor = scale * torch.eye(size, dtype=dtype, device=t.device)
                if isinstance(t, DTensor):
                    tensor = distribute_tensor(tensor, device_mesh=t.device_mesh, placements=t.placements)
                Q.append(tensor)

                piece1A.append(letters[i] + letters[i + 13])
                piece2A = piece2A + letters[i + 13]
                piece3A = piece3A + letters[i]

                piece1 = "".join([(letters[i + 13] if j == i else letters[j]) for j in range(len(shape))])
                piece2 = "".join([(letters[i + 26] if j == i else letters[j]) for j in range(len(shape))])
                subscripts = piece1 + "," + piece2 + "->" + letters[i + 13] + letters[i + 26]
                exprGs.append(subscripts)

                a, b, c = (letters[i], letters[i + 13], letters[i + 26])
                piece1P.append(a + b)
                piece2P.append(a + c)
                piece3P = piece3P + c
                piece4P = piece4P + b

        while len(Q) < 4:
            Q.append(-2)

        exprA = ",".join(piece1A) + "," + piece2A + "->" + piece3A
        exprP = ",".join(piece1P) + "," + ",".join(piece2P) + "," + piece3P + "->" + piece4P
    exprGs = tuple(exprGs)
    return [Q, (exprA, exprGs, exprP)]


# @torch.compile(fullgraph=True)
def _balance_Q(Q_in):
    # norms = torch.stack([torch.max(torch.abs(q)) for q in Q_in])
    norms = [torch.max(torch.abs(q)) for q in Q_in]
    geometric_mean = torch.stack(norms).prod() ** (1 / len(Q_in))
    for i, q in enumerate(Q_in):
        q.mul_(geometric_mean / norms[i])


def _lb(A, max_abs):
    A = A / max_abs
    aa = torch.real(A * A.conj())
    value0, i = torch.max(torch.sum(aa, dim=0), 0)
    value1, j = torch.max(torch.sum(aa, dim=1), 0)
    if value0 > value1:
        x = A[:, i].conj() @ A
        return max_abs * torch.linalg.vector_norm((x / torch.linalg.vector_norm(x)) @ A.H)
    else:
        x = A @ A[j].conj()
        return max_abs * torch.linalg.vector_norm(A.H @ (x / torch.linalg.vector_norm(x)))


def _norm_lower_bound(A):
    """Cheap lower bound for the spectral norm of A."""
    max_abs = A.norm(float("inf"))
    return torch.where(max_abs > 0, _lb(A, max_abs), max_abs)


def _solve_triangular_right(X, A):
    """X @ inv(A)"""
    orig_dtype = X.dtype
    X = X.to(dtype=torch.float32, non_blocking=True)
    A = A.to(dtype=torch.float32, non_blocking=True)
    return torch.linalg.solve_triangular(A, X[None, :], upper=True, left=False).to(dtype=orig_dtype, non_blocking=True)[
        0
    ]



# @torch.compile(fullgraph=True, dynamic=False)
def _calc_A_and_conjB(exprA, G, Q, V):
    A = torch.einsum(exprA, *Q, G)
    order = G.dim()
    p = list(range(order))
    conjB = torch.permute(V.conj(), p[1:] + p[:1])
    for i, q in enumerate(Q):
        conjB = conjB / q if q.dim() < 2 else _solve_triangular_right(conjB, q)
        if i < order - 1:
            conjB = torch.transpose(conjB, i, order - 1)
    return A, conjB


# @torch.compile(fullgraph=True, dynamic=False)
def _q_terms(exprGs, A, conjB):
    terms = []
    for exprG in exprGs:
        term1 = torch.einsum(exprG, A, A.conj())
        term2 = torch.einsum(exprG, conjB.conj(), conjB)
        terms.append((term1, term2))
    return terms


def _update_precond(Q, exprs, V, G, step, tiny):
    """Update Kronecker product preconditioner Q with pair (V, G)."""
    exprA, exprGs, _ = exprs

    A, conjB = _calc_A_and_conjB(exprA, G, Q, V)

    terms = _q_terms(exprGs, A, conjB)

    for q, (term1, term2) in zip(Q, terms):
        tmp = term1 - term2
        tmp *= step
        if q.dim() < 2:
            tmp *= q
            tmp /= (term1 + term2).norm(float("inf")) + tiny
            q.sub_(tmp)
        else:
            tmp = torch.triu(tmp)
            tmp /= _norm_lower_bound(term1 + term2) + tiny
            tmp @= q
            q.sub_(tmp)


# @torch.compile(fullgraph=True, dynamic=False)
def _precond_grad(Q, exprs, G):
    """Precondition gradient G with preconditioner Q."""
    return torch.einsum(exprs[-1], *[q.conj() for q in Q], *Q, G)
