"""
This code is based on code examples given in
[Elgendy M (2020) Deep Learning for Vision Systems. Manning Publications ].

Originally Keras was used for them and were adapted to pytorch.
"""

from sklearn.datasets import make_blobs
from matplotlib import pyplot
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.adamw import AdamW
from torchsummary import summary
import numpy as np
import tqdm
from datasets import Dataset

results_eval = list()
results_train = list()


loss_eval = list()
loss_train = list()


def train(
    model,
    dataloader: DataLoader,
    loss_fn,
    optimizer,
    progress_bar: tqdm.tqdm,
    device="cuda",
):
    num_batches = len(dataloader)
    dataset_size = len(dataloader.dataset)

    test_loss, correct = 0.0, 0.0

    model.train()
    for i, batch in enumerate(dataloader):
        data = batch["data"].to(device)
        target = batch["target"].to(device)

        prediction = model(data)
        loss = loss_fn(prediction, target)

        loss.backward()

        correct += (
            (
                np.argmax(F.softmax(prediction.detach(), -1).cpu(), -1)
                == np.argmax(target.cpu(), -1)
            )
            .type(torch.float)
            .sum()
            .item()
        )

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        test_loss += loss.item()

        # if i % reporting_interval == 0:
        #    print(f"\nCurrent loss {loss.item()}")
    correct /= dataset_size
    test_loss /= num_batches
    results_train.append(correct)
    loss_train.append(test_loss)


def eval(model, dataloader: DataLoader, loss_fn, device="cuda"):
    model.eval()

    size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0.0, 0.0

    progress_bar = tqdm.tqdm(range(num_batches))

    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            target = batch["target"].to(device)

            prediction = model(data)

            loss = loss_fn(prediction, target)
            test_loss = loss.item()

            correct += (
                (
                    np.argmax(F.softmax(prediction, -1).cpu(), -1)
                    == np.argmax(target.cpu(), -1)
                )
                .type(torch.float)
                .sum()
                .item()
            )

            progress_bar.update(1)

    test_loss  # /= num_batches
    correct /= size
    results_eval.append(correct)
    loss_eval.append(test_loss)
    print(
        f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    )


def collate_fn(batch):
    result = dict()
    result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([x["data"] for x in batch], dtype=torch.float32)

    return result


if __name__ == "__main__":
    n_samples = 1000

    # Create data
    X, y = make_blobs(
        n_samples=n_samples, centers=3, n_features=2, cluster_std=2, random_state=2
    )

    # One hot encode
    num_labels = max(y) + 1
    Y = np.array([[1 if x == n else 0 for x in range(num_labels)] for n in y])

    print(X)
    print(Y)
    n_train = int(n_samples * 0.8)

    train_X, test_X = X[:n_train], X[n_train:]
    train_Y, test_Y = Y[:n_train], Y[n_train:]

    train_dataset = Dataset.from_dict(
        {
            "data": train_X,
            "target": train_Y,
        }
    )

    test_dataset = Dataset.from_dict(
        {
            "data": test_X,
            "target": test_Y,
        }
    )

    print(train_dataset)
    print(train_dataset[0])

    train_dataloader = DataLoader(
        train_dataset,
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )
    test_dataloader = DataLoader(
        test_dataset,
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=4,
        persistent_workers=True,
    )

    print(train_X.shape, test_X.shape)

    model = nn.Sequential(
        nn.Linear(2, 25),
        nn.ReLU(),
        nn.LazyBatchNorm1d(),
        nn.Dropout(0.3),
        nn.Linear(25, 10),
        nn.ReLU(),
        nn.LazyBatchNorm1d(),
        nn.Dropout(0.3),
        nn.Linear(10, 3),
        # Loss function includes Softmax
        # nn.Softmax(),
    )
    optimizer = AdamW(model.parameters(), lr=1e-4)
    loss_fn = nn.CrossEntropyLoss()

    #    summary(model, (1, 2), device="cpu")

    epoch = 100

    progress_bar = tqdm.tqdm(range(epoch * len(train_dataloader)))

    model.to("cuda")
    for e in range(epoch):
        results_eval.clear()
        results_train.clear()

        loss_eval.clear()
        loss_train.clear()

        train(model, train_dataloader, loss_fn, optimizer, progress_bar)

        eval(model, test_dataloader, loss_fn)

    print("Finished")

    pyplot.figure()
    pyplot.plot(range(epoch), results_train, label="train")
    pyplot.plot(range(epoch), results_eval, label="eval")
    pyplot.title("Accuracy")
    pyplot.legend()

    pyplot.figure()
    pyplot.plot(range(epoch), loss_train, label="train")
    pyplot.plot(range(epoch), loss_eval, label="eval")
    pyplot.title("loss")
    pyplot.legend()

    pyplot.show()
