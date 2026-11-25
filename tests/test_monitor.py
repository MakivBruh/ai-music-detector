import math
import pytest
from hmf.monitor import TrainingMonitor


def test_patience_zero_runs_through_plateau():
    monitor = TrainingMonitor('macro_f1', patience=0)
    assert monitor.update({'macro_f1': 1 / 3}, 1)
    for epoch in range(2, 31):
        assert not monitor.update({'macro_f1': 1 / 3}, epoch)
        assert not monitor.should_stop
    assert monitor.best_epoch == 1


def test_original_plateau_stops_at_epoch_six():
    monitor = TrainingMonitor(patience=5)
    for epoch in range(1, 7):
        monitor.update({'macro_f1': 1 / 3}, epoch)
        assert monitor.should_stop == (epoch == 6)


def test_loss_selects_improved_epoch_with_tied_f1():
    monitor = TrainingMonitor('loss', patience=2, min_delta=.001)
    assert monitor.update({'loss': .69, 'macro_f1': 1 / 3}, 1)
    assert monitor.update({'loss': .67, 'macro_f1': 1 / 3}, 2)
    assert not monitor.update({'loss': .67, 'macro_f1': 1 / 3}, 3)
    assert not monitor.should_stop
    assert monitor.update({'loss': .65, 'macro_f1': 1 / 3}, 4)
    assert monitor.stale_epochs == 0
    assert monitor.best_epoch == 4


def test_invalid_settings_and_metrics():
    with pytest.raises(ValueError):
        TrainingMonitor(patience=-1)
    with pytest.raises(ValueError):
        TrainingMonitor(min_delta=-1)
    with pytest.raises(ValueError):
        TrainingMonitor().update({'macro_f1': math.nan}, 1)
