import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyValueNet(nn.Module):
    def __init__(self, obs_dim=144, act_dim=12):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, 512)
        self.norm1 = nn.LayerNorm(512)
        self.fc2 = nn.Linear(512, 1024)
        self.norm2 = nn.LayerNorm(1024)
        self.fc3 = nn.Linear(1024, 512)
        self.norm3 = nn.LayerNorm(512)
        self.fc4 = nn.Linear(512, 128)
        self.norm4 = nn.LayerNorm(128)
        self.policy_head = nn.Linear(128, act_dim)
        self.value_head = nn.Linear(128, 1)


        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        h = F.relu(self.norm1(self.fc1(x)))
        h = F.relu(self.norm2(self.fc2(h)))
        h = F.relu(self.norm3(self.fc3(h)))
        h = F.relu(self.norm4(self.fc4(h)))

        policy_logits = self.policy_head(h)
        value = torch.tanh(self.value_head(h))
        return policy_logits, value


    @torch.no_grad()
    def predict(self, x):
        policy_logits, value = self.forward(x)
        policy = F.softmax(policy_logits, dim=-1)
        return policy.squeeze(0), value.squeeze(0)
