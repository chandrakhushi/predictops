import numpy as np

from predictops.features import NAMES, W, window_features
from predictops.train import main


def test_slope_and_time_to_exhaustion():
    series = np.zeros((W, 6), np.float32)
    series[:, 1] = 50 + 0.5 * np.arange(W)  # memory climbing 0.5%/tick
    f = dict(zip(NAMES, window_features(series)[0]))
    assert abs(f["mem_slope"] - 0.5) < 1e-4
    assert abs(f["mem_tte"] - (98 - f["mem_last"]) / 0.5) < 1e-2


def test_model_warns_earlier_than_static_thresholds(tmp_path):
    results = main(hosts=16, days=4, out=str(tmp_path / "m.joblib"))
    static, model = results.values()
    assert model["caught"] >= 0.8
    assert model["median_lead_min"] > static["median_lead_min"]
