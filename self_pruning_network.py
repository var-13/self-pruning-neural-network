"""
Self-Pruning Neural Network
Tredence AI Engineering Case Study

Implements learnable gating with L1 sparsity regularization.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torchvision
import torchvision.transforms as transforms

# ---------------- DEVICE ----------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------- PRUNABLE LAYER ----------------
class PrunableLinear(nn.Module):
    """
    Custom linear layer with learnable gates.
    Each weight is multiplied by sigmoid(gate_score).
    """

    def __init__(self, in_f, out_f):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_f, in_f) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_f))
        self.gate_scores = nn.Parameter(torch.zeros(out_f, in_f))

    def forward(self, x):
        gates = torch.sigmoid(self.gate_scores)
        pruned_weights = self.weight * gates
        return F.linear(x, pruned_weights, self.bias)


# ---------------- MODEL ----------------
class Net(nn.Module):
    """
    Feedforward network using PrunableLinear layers.
    """

    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.fc1 = PrunableLinear(3072, 256)
        self.fc2 = PrunableLinear(256, 128)
        self.fc3 = PrunableLinear(128, 10)

    def forward(self, x):
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

    def sparsity_loss(self):
        """
        L1 penalty on gate values to encourage pruning.
        """
        gates = torch.cat([
            torch.sigmoid(self.fc1.gate_scores).flatten(),
            torch.sigmoid(self.fc2.gate_scores).flatten(),
            torch.sigmoid(self.fc3.gate_scores).flatten()
        ])
        return gates.mean()

    def sparsity(self):
        """
        Measures sparsity based on effective weights (w * gate).
        """
        weights = torch.cat([
            (self.fc1.weight * torch.sigmoid(self.fc1.gate_scores)).abs().flatten(),
            (self.fc2.weight * torch.sigmoid(self.fc2.gate_scores)).abs().flatten(),
            (self.fc3.weight * torch.sigmoid(self.fc3.gate_scores)).abs().flatten()
        ])
        return (weights < 1e-3).float().mean().item()


# ---------------- DATA ----------------
def get_data():
    """
    Loads CIFAR-10 dataset (subset for faster training).
    """
    transform = transforms.Compose([transforms.ToTensor()])

    train = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform
    )
    test = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform
    )

    # Use subset for faster execution
    train = Subset(train, range(5000))
    test = Subset(test, range(1000))

    train_loader = DataLoader(train, batch_size=256, shuffle=True)
    test_loader = DataLoader(test, batch_size=256)

    return train_loader, test_loader


# ---------------- TRAIN ----------------
def train_model(lam):
    """
    Trains model for a given lambda value.
    """
    model = Net().to(device)
    train_loader, test_loader = get_data()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Training loop
    for epoch in range(15):
        model.train()
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            outputs = model(x)

            loss = F.cross_entropy(outputs, y) + lam * model.sparsity_loss()
            loss.backward()
            optimizer.step()

    # Evaluation
    model.eval()
    correct = total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    acc = correct / total
    sp = model.sparsity()

    print(f"λ={lam} | acc={acc:.2f} | sparsity={sp:.2f}")


# ---------------- MAIN ----------------
if __name__ == "__main__":
    for lam in [1e-3, 3e-3, 5e-3]:
        train_model(lam)
