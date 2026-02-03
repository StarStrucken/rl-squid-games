import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyGradientFF(nn.Module):
    def __init__(
        self,
        n_features,
        n_actions,
        neurons=[64, 64],
        activation_function=F.relu,
        learning_rate=3e-4,
    ):
        super().__init__()

        self.activation = activation_function

        # shared backbone
        layers = []
        in_dim = n_features
        for h in neurons:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        self.backbone = nn.Sequential(*layers)

        # policy head
        self.policy_head = nn.Linear(in_dim, n_actions)

        # value head (baseline)
        self.value_head = nn.Linear(in_dim, 1)

        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    def forward(self, x):
        x = x.float()
        h = self.backbone(x)
        policy = F.softmax(self.policy_head(h), dim=1)
        value = self.value_head(h).squeeze(1)
        return policy, value

    def get_loss(self, states, actions, returns, entropy_coef=0.003, value_coef=0.5):
        states = torch.tensor(states).float()
        actions = torch.tensor(actions).long()
        returns = torch.tensor(returns).float()

        policy, values = self.forward(states)

        # advantage = return - baseline
        advantages = returns - values.detach()

        # log probabilities
        logp = torch.log(policy + 1e-8)
        chosen_logp = logp.gather(1, actions.unsqueeze(1)).squeeze(1)

        # policy loss
        policy_loss = -(advantages * chosen_logp).mean()

        # value loss (MSE)
        value_loss = F.mse_loss(values, returns)

        # entropy bonus
        entropy = -(policy * logp).sum(dim=1).mean()

        total_loss = (
            policy_loss
            + value_coef * value_loss
            - entropy_coef * entropy
        )

        return total_loss

    def custom_train(self, states, actions, returns, epochs=1):
        for _ in range(epochs):
            self.optimizer.zero_grad()
            loss = self.get_loss(states, actions, returns)
            loss.backward()
            self.optimizer.step()
