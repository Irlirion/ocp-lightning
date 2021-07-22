from typing import Optional

from models.energy_model import EnergyModel
from models.modules.edge_update_net import EdgeUpdateNetWrap


class EdgeUpdateNetWrapper(EnergyModel):
    def __init__(
        self,
        lr: int,
        weight_decay: float,
        normalize_labels: bool,
        mean: Optional[float],
        std: Optional[float],
        hidden_channels=64,
        num_interactions=3,
        cutoff=6,
    ):
        super().__init__(
            EdgeUpdateNetWrap(
                cutoff=cutoff, 
                regress_forces=False, 
                hidden_channels=hidden_channels,
                num_interactions=num_interactions,
            ),
            lr=lr,
            weight_decay=weight_decay,
            normalize_labels=normalize_labels,
            mean=mean,
            std=std,
        )