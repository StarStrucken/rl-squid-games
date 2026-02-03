import torch
import os
import torch.nn as nn
import numpy as np
from sai_rl import SAIClient
from policy_gradient import PolicyGradientFF

sai = SAIClient(comp_id="squid-hunt-ualberta", api_key=os.environ["SAI_API_KEY"])
env = sai.make_env()

class PolicyOnlyModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.base = base_model

    def forward(self, x):
        if x.ndim == 1:
            x = x.unsqueeze(0)

        out = self.base(x)

        if hasattr(out, "__len__") and len(out) == 2:
            out = out[0]

        return out


model = PolicyGradientFF(
    n_features=env.observation_space.shape[0],  # type: ignore
    n_actions=env.action_space.n,               # type: ignore
    neurons=[64, 64],
    learning_rate=3e-4,
)

state_dict = torch.load("model.pt", map_location="cpu")
model.load_state_dict(state_dict)
model.eval()
model.to("cpu")

def action_function(policy):
    # try to unpack (policy, value); if it fails, policy is already correct
    try:
        p = policy[0]
    except:
        p = policy

    # convert torch tensor → numpy
    try:
        p = p.detach().cpu().numpy()
    except:
        pass

    p = np.asarray(p)

    # single observation
    if p.ndim == 1:
        return int(np.argmax(p))

    # return batch
    return np.argmax(p, axis=1)


submit_model = PolicyOnlyModel(model).eval().to("cpu")

sai.submit("My Model (loaded)", submit_model, action_function)