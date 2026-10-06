"""Train the PyTorch model with a hand-written training loop.

Run:  python -m src.train_nn
"""
import copy
import time

import joblib
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from src.config import MODELS, RANDOM_SEED
from src.evaluate import evaluate, save_metrics
from src.nn_model import FraudDataset, TabularMLP
from src.preprocess import Preprocessor, load_splits, target

BATCH_SIZE = 2048
MAX_EPOCHS = 30
PATIENCE = 4          # stop after this many epochs without validation improvement
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-5


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader) -> np.ndarray:
    model.eval()  # switches Dropout off and BatchNorm to its running statistics
    scores = [torch.sigmoid(model(x_num, x_cat)) for x_num, x_cat, _ in loader]
    return torch.cat(scores).numpy()


def main() -> None:
    torch.manual_seed(RANDOM_SEED)
    train, valid, test = load_splits()

    # Reuse the preprocessor fitted on train by train_gbm, so all models see
    # exactly the same encoding.
    pre: Preprocessor = joblib.load(MODELS / "preprocessor.joblib")
    datasets = {}
    for name, df in [("train", train), ("valid", valid), ("test", test)]:
        x_num, x_cat = pre.transform_dense(df)
        datasets[name] = FraudDataset(x_num, x_cat, target(df))

    train_loader = DataLoader(datasets["train"], batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(datasets["valid"], batch_size=8192)
    test_loader = DataLoader(datasets["test"], batch_size=8192)

    model = TabularMLP(n_numeric=datasets["train"].x_num.shape[1],
                       cardinalities=pre.cardinalities())
    print(model)
    print(f"trainable parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Class imbalance: fraud is ~3.5% of rows. pos_weight makes each fraud
    # example count more in the loss. The full ratio (~27) over-corrects and
    # makes training noisy; its square root (~5) is a common compromise.
    y_train = datasets["train"].y
    pos_weight = torch.sqrt((y_train == 0).sum() / (y_train == 1).sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    best_pr_auc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(1, MAX_EPOCHS + 1):
        start = time.time()
        model.train()  # Dropout on, BatchNorm uses batch statistics
        total_loss = 0.0
        for x_num, x_cat, y in train_loader:
            optimizer.zero_grad()           # 1. clear gradients from the previous step
            logits = model(x_num, x_cat)    # 2. forward pass
            loss = loss_fn(logits, y)       # 3. how wrong are we?
            loss.backward()                 # 4. backpropagation: d(loss)/d(weight)
            optimizer.step()                # 5. update weights
            total_loss += loss.item() * len(y)

        val_metrics = evaluate(datasets["valid"].y.numpy(), predict(model, valid_loader))
        print(f"epoch {epoch:2d} | train loss {total_loss / len(datasets['train']):.4f} | "
              f"valid PR-AUC {val_metrics['pr_auc']:.4f} | {time.time() - start:.0f}s")

        # Early stopping: keep the weights from the best validation epoch.
        if val_metrics["pr_auc"] > best_pr_auc:
            best_pr_auc = val_metrics["pr_auc"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= PATIENCE:
                print(f"early stopping, best valid PR-AUC {best_pr_auc:.4f}")
                break

    model.load_state_dict(best_state)
    torch.save(
        {"state_dict": model.state_dict(),
         "n_numeric": datasets["train"].x_num.shape[1],
         "cardinalities": pre.cardinalities()},
        MODELS / "pytorch_mlp.pt",
    )
    test_scores = predict(model, test_loader)
    np.save(MODELS / "test_scores_pytorch.npy", test_scores)
    save_metrics("pytorch", evaluate(datasets["test"].y.numpy(), test_scores))


if __name__ == "__main__":
    main()
