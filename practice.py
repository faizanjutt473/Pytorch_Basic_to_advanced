"""
PyTorch: Simple se Advanced (6 Levels)
Run: python pytorch_simple_to_advanced.py
Har level ko alag alag samajh kar run karein (neeche main() mein comment/uncomment kar sakte hain).
"""
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset, random_split

torch.manual_seed(42)
device = "cuda" if torch.cuda.is_available() else "cpu"


# ------------------------------------------------------------------
# LEVEL 1: Tensors (PyTorch ki buniyad)
# ------------------------------------------------------------------
def level1_tensors():
    print("\n=== LEVEL 1: Tensors ===")
    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    b = torch.ones(2, 2)
    print("a + b =\n", a + b)
    print("a * b (element-wise) =\n", a * b)
    print("a @ b (matrix mult) =\n", a @ b)
    print("shape:", a.shape, "| dtype:", a.dtype)
    print("reshape:", a.reshape(4), "| mean:", a.mean().item())


# ------------------------------------------------------------------
# LEVEL 2: Autograd (automatic gradients)
# ------------------------------------------------------------------
def level2_autograd():
    print("\n=== LEVEL 2: Autograd ===")
    x = torch.tensor(3.0, requires_grad=True)
    y = x ** 2 + 2 * x + 1      # y = x^2 + 2x + 1
    y.backward()                # dy/dx = 2x + 2 = 8
    print("dy/dx at x=3:", x.grad.item())


# ------------------------------------------------------------------
# LEVEL 3: Linear Regression (manual, bina nn.Module ke)
# ------------------------------------------------------------------
def level3_manual_linear_regression():
    print("\n=== LEVEL 3: Manual Linear Regression ===")
    X = torch.linspace(0, 1, 100).unsqueeze(1)
    y = 3 * X + 2 + 0.1 * torch.randn_like(X)   # true: w=3, b=2

    w = torch.randn(1, requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    lr = 0.1

    for epoch in range(300):
        pred = X * w + b
        loss = ((pred - y) ** 2).mean()
        loss.backward()
        with torch.no_grad():
            w -= lr * w.grad
            b -= lr * b.grad
        w.grad.zero_()
        b.grad.zero_()
    print(f"Learned w={w.item():.2f}, b={b.item():.2f} (true: 3, 2)")


# ------------------------------------------------------------------
# LEVEL 4: nn.Module + Optimizer + Loss (standard tareeqa)
# ------------------------------------------------------------------
class LinearModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(1, 1)

    def forward(self, x):
        return self.linear(x)


def level4_nn_module():
    print("\n=== LEVEL 4: nn.Module ===")
    X = torch.linspace(0, 1, 100).unsqueeze(1)
    y = 3 * X + 2 + 0.1 * torch.randn_like(X)

    model = LinearModel()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    for epoch in range(300):
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        optimizer.step()
    w, b = model.linear.weight.item(), model.linear.bias.item()
    print(f"Learned w={w:.2f}, b={b:.2f}")


# ------------------------------------------------------------------
# LEVEL 5: Classification (MLP + DataLoader + mini-batches + accuracy)
# ------------------------------------------------------------------
def level5_classification():
    print("\n=== LEVEL 5: Binary Classification (MLP) ===")
    n = 500
    X = torch.cat([torch.randn(n, 2) + 1.5, torch.randn(n, 2) - 1.5])
    y = torch.cat([torch.ones(n), torch.zeros(n)]).unsqueeze(1)

    loader = DataLoader(TensorDataset(X, y), batch_size=64, shuffle=True)

    model = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 8), nn.ReLU(),
        nn.Linear(8, 1),            # logits (sigmoid loss ke andar hai)
    )
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(20):
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X)) > 0.5).float()
        acc = (preds == y).float().mean().item()
    print(f"Accuracy: {acc:.2%}")


# ------------------------------------------------------------------
# LEVEL 6 (ADVANCED): Custom Dataset + ResNet-style CNN + full training
# pipeline: train/val split, GPU, AMP, scheduler, grad clipping,
# early stopping, best-model checkpoint, test evaluation
# ------------------------------------------------------------------
class QuadrantDataset(Dataset):
    """Synthetic images: class = kaun sa quadrant sab se roshan hai (4 classes)."""

    def __init__(self, n=2000, size=28):
        self.images = torch.randn(n, 1, size, size) * 0.5
        self.labels = torch.randint(0, 4, (n,))
        h = size // 2
        slices = [(slice(0, h), slice(0, h)), (slice(0, h), slice(h, size)),
                  (slice(h, size), slice(0, h)), (slice(h, size), slice(h, size))]
        for i, lbl in enumerate(self.labels):
            r, c = slices[lbl]
            self.images[i, 0, r, c] += 1.0

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


class ResBlock(nn.Module):
    """Skip connection wala block: out = F(x) + x"""

    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + x)


class SmallResNet(nn.Module):
    def __init__(self, num_classes=4):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1, bias=False),
            nn.BatchNorm2d(16), nn.ReLU(),
        )
        self.block1 = ResBlock(16)
        self.down = nn.Sequential(nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU())
        self.block2 = ResBlock(32)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down(x)
        x = self.block2(x)
        x = self.pool(x).flatten(1)
        return self.fc(self.dropout(x))


def run_epoch(model, loader, criterion, optimizer=None, scaler=None, use_amp=False):
    """Ek epoch chalata hai. optimizer diya to train mode, warna eval mode."""
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0

    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            with torch.autocast(device_type=device, enabled=use_amp):
                logits = model(xb)
                loss = criterion(logits, yb)

            if training:
                optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()

            total_loss += loss.item() * xb.size(0)
            correct += (logits.argmax(1) == yb).sum().item()
            total += xb.size(0)
    return total_loss / total, correct / total


def level6_advanced():
    print(f"\n=== LEVEL 6: Advanced Pipeline (device: {device}) ===")
    dataset = QuadrantDataset(n=3000)
    train_set, val_set, test_set = random_split(dataset, [2000, 500, 500])
    train_loader = DataLoader(train_set, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=128)
    test_loader = DataLoader(test_set, batch_size=128)

    model = SmallResNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    epochs = 15
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    use_amp = device == "cuda"
    scaler = torch.amp.GradScaler(device, enabled=use_amp)

    best_val, best_state, patience, bad_epochs = float("inf"), None, 3, 0

    for epoch in range(1, epochs + 1):
        tr_loss, tr_acc = run_epoch(model, train_loader, criterion, optimizer, scaler, use_amp)
        va_loss, va_acc = run_epoch(model, val_loader, criterion, use_amp=use_amp)
        scheduler.step()
        print(f"Epoch {epoch:2d} | train loss {tr_loss:.3f} acc {tr_acc:.2%} "
              f"| val loss {va_loss:.3f} acc {va_acc:.2%}")

        # Best model save + early stopping
        if va_loss < best_val:
            best_val, bad_epochs = va_loss, 0
            best_state = copy.deepcopy(model.state_dict())
            torch.save(best_state, "best_model.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping!")
                break

    model.load_state_dict(best_state)
    te_loss, te_acc = run_epoch(model, test_loader, criterion, use_amp=use_amp)
    print(f"\nTEST accuracy: {te_acc:.2%}")


if __name__ == "__main__":
    level1_tensors()
    level2_autograd()
    level3_manual_linear_regression()
    level4_nn_module()
    level5_classification()
    level6_advanced()
