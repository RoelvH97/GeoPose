import torch
import torch.nn as nn

from geopose.inference import TestTimeOptimizer as BaseTTO


class QuadraticOptimizer(BaseTTO):
    def __init__(self):
        nn.Module.__init__(self)
        self.rotations = nn.ParameterDict(
            {
                "lat": nn.Parameter(torch.tensor([[2.0]])),
                "pa": nn.Parameter(torch.tensor([[-2.0]])),
            }
        )
        self.translations_scaled = nn.ParameterDict(
            {
                "lat": nn.Parameter(torch.tensor([[0.0]])),
                "pa": nn.Parameter(torch.tensor([[0.0]])),
            }
        )
        self.multiplier = 1.0

    def pose(self, view):
        return self.rotations[view], self.translations_scaled[view]

    def losses(self):
        targets = {"lat": 0.25, "pa": -0.5}
        ncc = {
            view: ((self.rotations[view] - target) ** 2).mean()
            for view, target in targets.items()
        }
        dice = {
            view: (self.translations_scaled[view] ** 2).mean()
            for view in ("lat", "pa")
        }
        return ncc, dice, sum(ncc.values()) + sum(dice.values())


def test_tto_trace_and_per_view_best_selection():
    optimizer = QuadraticOptimizer()
    _, trace = optimizer.optimize(iterations=5)
    assert trace["optimizer"] == "NAdam"
    assert trace["scheduler"] == "OneCycleLR"
    assert trace["pct_start"] == 0.3
    assert len(trace["steps"]) == 6
    assert [item["step"] for item in trace["steps"]] == list(range(6))
    for view in ("lat", "pa"):
        best = [item["views"][view]["best_mncc"] for item in trace["steps"]]
        assert all(right >= left for left, right in zip(best, best[1:]))


