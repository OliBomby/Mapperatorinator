import numpy as np
import torch


class Averager:
    def __init__(self):
        self.reset()

    def reset(self):
        self.total_sum = {}
        self.count = {}

    def update(self, stats: dict):
        for key, value in stats.items():
            if isinstance(value, torch.Tensor):
                current_val = value.sum().item()
                current_n = value.numel()
            elif isinstance(value, np.ndarray):
                current_val = value.sum()
                current_n = value.size
            else:
                current_val = value
                current_n = 1

            if key not in self.total_sum:
                self.total_sum[key] = current_val
                self.count[key] = current_n
            else:
                self.total_sum[key] += current_val
                self.count[key] += current_n
    def average(self):
        averaged_stats = {
            key: (tot / self.count[key]).item() if isinstance(tot, torch.Tensor) else tot / self.count[key] for key, tot in self.total_sum.items()
        }
        self.reset()

        return averaged_stats   
    def get_current_averages(self) -> dict: # Renamed from get_latest_averages
        if not self.count: return {}
        averaged_stats = {}
        for key, total_val in self.total_sum.items():
            count_val = self.count.get(key, 0)
            if count_val > 0: averaged_stats[key] = total_val / count_val
            else: averaged_stats[key] = float('nan')
        return averaged_stats

    def get_final_averages(self) -> dict:
        averaged_stats = self.get_current_averages()
        self.reset()
        return averaged_stats