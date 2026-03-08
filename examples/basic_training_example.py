import os
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from torchdiag import TorchDiag


def main() -> None:
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inputs = torch.randn(512, 32)
    targets = torch.randint(0, 2, (512,))
    dataset = TensorDataset(inputs, targets)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    model = nn.Sequential(nn.Linear(32, 64), nn.ReLU(), nn.Linear(64, 2)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    diag = TorchDiag(
        job_name="synthetic_demo",
        sample_interval_ms=200,
        snapshot_interval_steps=5,
    )

    diag.start()
    loader_iter = iter(loader)
    for step in range(10):
        with diag.event("dataloader_wait", step=step):
            batch = next(loader_iter)
        x, y = batch[0].to(device), batch[1].to(device)
        with diag.step_context(step=step, batch_size=x.size(0)):
            with diag.event("forward"):
                outputs = model(x)
                loss = loss_fn(outputs, y)
            with diag.event("backward"):
                loss.backward()
            with diag.event("optimizer"):
                optimizer.step()
                optimizer.zero_grad()
        time.sleep(0.01)

    diag.stop()
    os.makedirs("artifacts", exist_ok=True)
    diag.write_json("artifacts/run.json")
    diag.write_html("artifacts/report.html")


if __name__ == "__main__":
    main()
