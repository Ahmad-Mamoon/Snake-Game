import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

MODEL_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "model",
)


class Linear_QNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = self.linear2(x)
        return x

    def save(self, file_name="model.pth", record=None):
        os.makedirs(MODEL_DIR, exist_ok=True)
        torch.save(self.state_dict(), os.path.join(MODEL_DIR, file_name))
        if record is not None:
            with open(os.path.join(MODEL_DIR, file_name + ".record"), "w") as f:
                f.write(str(record))

    @staticmethod
    def saved_record(file_name="model.pth"):
        try:
            with open(os.path.join(MODEL_DIR, file_name + ".record")) as f:
                return int(f.read().strip())
        except (OSError, ValueError):
            return 0

    def load(self, file_name="model.pth"):
        path = os.path.join(MODEL_DIR, file_name)
        if os.path.exists(path):
            self.load_state_dict(torch.load(path))
            self.eval()
            return True
        return False


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.lr = lr
        self.gamma = gamma
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.lr)
        self.criterion = nn.MSELoss()

    def train_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = torch.unsqueeze(state, 0)
            next_state = torch.unsqueeze(next_state, 0)
            action = torch.unsqueeze(action, 0)
            reward = torch.unsqueeze(reward, 0)
            done = (done,)

        pred = self.model(state)
        target = pred.clone()
        for idx in range(len(done)):
            Q_new = reward[idx]
            if not done[idx]:
                Q_new = reward[idx] + self.gamma * torch.max(self.model(next_state[idx]))
            target[idx][torch.argmax(action[idx]).item()] = Q_new

        self.optimizer.zero_grad()
        loss = self.criterion(target, pred)
        loss.backward()
        self.optimizer.step()
