import math
import torch
from torch.optim import Optimizer

class AdEMAMix(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999, 0.9999), alpha=5.0, eps=1e-8, weight_decay=0.0, T_alpha=0, T_beta3=0):
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay, alpha=alpha, T_alpha=T_alpha, T_beta3=T_beta3)
        super(AdEMAMix, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p.data)  # Fast EMA
                    state['m2'] = torch.zeros_like(p.data)  # Slow EMA
                    state['v'] = torch.zeros_like(p.data)   # Second moment (like ADAM)

                m1, m2, v = state['m1'], state['m2'], state['v']
                beta1, beta2, beta3_final = group['betas']
                eps, alpha_final = group['eps'], group['alpha']
                lr, weight_decay = group['lr'], group['weight_decay']
                T_alpha, T_beta3 = group['T_alpha'], group['T_beta3']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                # Schedulers for alpha and beta3
                alpha = alpha_scheduler(state['step'], alpha_final, T_alpha)
                beta3 = beta3_scheduler(state['step'], beta1, beta3_final, T_beta3)

                # Update fast EMA
                m1.mul_(beta1).add_(1 - beta1, grad)
                
                # Update slow EMA
                m2.mul_(beta3).add_(1 - beta3, grad)

                # Update second moment estimate (similar to ADAM)
                v.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                # Compute bias-corrected first moment estimate
                m1_hat = m1 / bias_correction1

                # Compute bias-corrected second moment estimate
                v_hat = v / bias_correction2

                # Parameter update step
                denom = (v_hat.sqrt() + eps)
                update = (m1_hat + alpha * m2) / denom

                if weight_decay != 0:
                    update.add_(p.data, alpha=weight_decay)

                p.data.add_(-lr * update)

        return loss

# Schedulers for alpha and beta3 based on training steps
def alpha_scheduler(step, alpha_final, T_alpha):
    if T_alpha == 0:
        return alpha_final
    return min(step / T_alpha, 1.0) * alpha_final

def beta3_scheduler(step, beta_start, beta3_final, T_beta3):
    if T_beta3 == 0:
        return beta3_final
    return beta_start + (beta3_final - beta_start) * min(step / T_beta3, 1.0)