from torch.optim.lr_scheduler import _LRScheduler


class WarmupPolyLRScheduler(_LRScheduler):
    def __init__(
        self,
        optimizer,
        T_max,
        warmup_iters=500,
        power=0.9,
        eta_min=5e-6,
        last_epoch=-1,
    ):
        self.T_max = float(T_max)
        self.warmup_iters = float(warmup_iters)
        self.power = power
        self.eta_min = eta_min
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        current_iter = float(self.last_epoch)

        if current_iter < self.warmup_iters:
            return [
                base_lr * (current_iter / self.warmup_iters)
                for base_lr in self.base_lrs
            ]

        lr_ratio = (1.0 - current_iter / self.T_max) ** self.power
        return [
            self.eta_min + (base_lr - self.eta_min) * lr_ratio
            for base_lr in self.base_lrs
        ]
