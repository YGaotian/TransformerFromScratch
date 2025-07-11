import math
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim


def warmup_cosine_annealing(
        optimizer: optim.Optimizer,
        warmup_steps: int,
        all_training_steps: int,
        cos_period_num: float = 0.5,
        last_epoch: int = -1
):
    assert warmup_steps <= all_training_steps, "Training steps are not enough to complete a warmup cycle."
    assert all_training_steps > 0, "Training steps must be greater than 0."

    def lr_multiplier(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(warmup_steps)
        else:
            cos_period = math.pi * 2.0
            after_warm_steps = current_step - warmup_steps
            percentage = float(after_warm_steps) / (all_training_steps -
                                                    warmup_steps) if after_warm_steps else after_warm_steps
            # (1 + cos(x)) / 2 = [0, 1]
            decay_rate = (1 + math.cos(percentage * cos_period * cos_period_num)) * 0.5
            return decay_rate

    return LambdaLR(optimizer, lr_multiplier, last_epoch)
