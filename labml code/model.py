import torch
from torch import nn


INPUT_DIM = 12
N_ACTIONS = 4

class Model(nn.Module):

    def __init__(self):
        super().__init__()

        # This head gives the state value $V$
        self.state_value = nn.Sequential(
            nn.Linear(INPUT_DIM, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )
        # This head gives the action value $A$
        self.action_value = nn.Sequential(
            nn.Linear(INPUT_DIM, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, N_ACTIONS),
        )

    def forward(self, x: torch.Tensor):
        # $A$
        action_value = self.action_value(x)
        # $V$
        state_value = self.state_value(x)

        # $A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')$
        action_score_centered = action_value - action_value.mean(dim=-1, keepdim=True)
        # $Q(s, a) =V(s) + \Big(A(s, a) - \frac{1}{|\mathcal{A}|} \sum_{a' \in \mathcal{A}} A(s, a')\Big)$
        q = state_value + action_score_centered

        return q