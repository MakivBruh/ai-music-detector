"""Checkpoint selection and early stopping using validation metrics only."""
import math


class TrainingMonitor:
    def __init__(self, metric='macro_f1', patience=5, min_delta=0.0):
        if metric not in ('macro_f1', 'loss'):
            raise ValueError('monitor must be macro_f1 or loss')
        if patience < 0 or not math.isfinite(min_delta) or min_delta < 0:
            raise ValueError('patience and min_delta must be nonnegative')
        self.metric = metric
        self.patience = patience
        self.min_delta = min_delta
        self.best_value = None
        self.best_epoch = None
        self.stale_epochs = 0

    def update(self, metrics, epoch):
        value = float(metrics[self.metric])
        if not math.isfinite(value):
            raise ValueError('Non-finite validation metric')
        improvement = (self.best_value - value if self.metric == 'loss'
                       else value - self.best_value) if self.best_value is not None else math.inf
        improved = improvement > self.min_delta
        if improved:
            self.best_value, self.best_epoch, self.stale_epochs = value, epoch, 0
        else:
            self.stale_epochs += 1
        return improved

    @property
    def should_stop(self):
        return self.patience > 0 and self.stale_epochs >= self.patience
