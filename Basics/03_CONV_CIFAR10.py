"""
This code is based on code examples given in
[Elgendy M (2020) Deep Learning for Vision Systems. Manning Publications ].

Originally Keras was used for them and were adapted to pytorch.
"""

import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
import torch.nn as nn
import datasets
from torchsummary import summary
import tqdm
from torchvision.transforms import v2

results_train = list()
loss_train = list()

results_eval = list()
loss_eval = list()


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

        optimizer.step()
        optimizer.zero_grad()
        progress_bar.update(1)

        correct += (
            (
                np.argmax(F.softmax(prediction.detach(), -1).cpu(), -1)
                == np.argmax(target.cpu(), -1)
            )
            .type(torch.float)
            .sum()
            .item()
        )

        test_loss += loss.item()

    correct /= dataset_size
    test_loss /= num_batches
    results_train.append(correct)
    loss_train.append(test_loss)


def eval(model, dataloader: DataLoader, loss_fn, device="cuda"):
    model.eval()

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0.0, 0.0

    progress_bar = tqdm.tqdm(range(num_batches))

    with torch.no_grad():
        for batch in dataloader:
            data = batch["data"].to(device)
            target = batch["target"].to(device)

            prediction = model(data)

            loss = loss_fn(prediction, target)
            test_loss += loss.item()

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

    correct /= dataset_size
    test_loss /= num_batches
    results_eval.append(correct)
    loss_eval.append(test_loss)

    print(f"\nTest Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f} \n")


def collate_fn(batch):
    result = dict()
    result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([x["data"] for x in batch], dtype=torch.float32)

    return result


idx_to_label = {
    0: "airplane",
    1: "automobile",
    2: "bird",
    3: "cat",
    4: "deer",
    5: "dog",
    6: "frog",
    7: "horse",
    8: "ship",
    9: "truck",
}

if __name__ == "__main__":
    epoch = 10

    input_channels = 3
    base_hidden_units = 32

    model = nn.Sequential(
        # 32x32
        # Conv 1
        nn.Conv2d(input_channels, base_hidden_units, (3, 3), 1, padding="same"),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Conv 2
        nn.Conv2d(base_hidden_units, base_hidden_units, (3, 3), 1, padding="same"),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Pool + Dropout
        nn.MaxPool2d((2, 2), 2),
        nn.Dropout2d(0.2),
        # 16x16
        # Conv 3
        nn.Conv2d(base_hidden_units, base_hidden_units * 2, (3, 3), 1, padding="same"),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Conv 4
        nn.Conv2d(
            base_hidden_units * 2, base_hidden_units * 2, (3, 3), 1, padding="same"
        ),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Pool + Dropout
        nn.MaxPool2d((2, 2), 2),
        nn.Dropout2d(0.3),
        # 8x8
        # Conv 5
        nn.Conv2d(
            base_hidden_units * 2, base_hidden_units * 4, (3, 3), 1, padding="same"
        ),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Conv 6
        nn.Conv2d(
            base_hidden_units * 4, base_hidden_units * 4, (3, 3), 1, padding="same"
        ),
        nn.ReLU(),
        nn.LazyBatchNorm2d(),
        # Pool + Dropout
        nn.MaxPool2d((2, 2), 2),
        nn.Dropout2d(0.4),
        # 4 x 4
        # Flatten the feature maps
        nn.Flatten(),
        # Classification layers
        nn.LazyLinear(10),
    )

    if os.path.exists("model.pt"):
        model = torch.load("model.pt", weights_only=False)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LinearLR(optimizer, total_iters=epoch)

    summary(model, (3, 32, 32), device="cpu")

    # Prepare data
    dataset_raw = datasets.load_dataset("cifar10")

    dataset_onehot = dataset_raw.map(
        lambda x: {"target": [0 if v != x["label"] else 1 for v in range(10)]}
    )

    # Convert Image to normalized representation
    # Because it is colored and the color channels are the last dimension
    # it needs to be transposed to make into three maps each representing a color
    dataset_converted_image = dataset_onehot.map(
        lambda x: {
            "data": torch.from_numpy(
                np.array(x["img"]).astype(np.float32),
            )
            .transpose(0, -1)
            .transpose(1, 2)
        }
    )

    # Mean precalculated because otherwise it will take a long time
    mean = 120.70756512369792  # np.mean(dataset_converted_image["train"]["img"])
    std = 64.1500758911212  # np.std(dataset_converted_image["train"]["img"])

    dataset_normalized_image = dataset_converted_image.map(
        lambda x: {"data": (np.array(x["data"]) - mean) / std}
    )

    dataset = dataset_normalized_image.remove_columns("img")
    dataset = dataset.remove_columns("label")

    print(dataset)
    dataset_validation = dataset["train"].train_test_split(0.1)
    dataset["train"] = dataset_validation["train"]
    dataset["valid"] = dataset_validation["test"]

    data_transform = {
        "train": v2.Compose(
            [
                v2.RandomRotation(15),
                v2.RandomHorizontalFlip(0.1),
                v2.RandomAffine(0, (0.2, 0.2)),
            ]
        ),
        # "valid": v2.Compose(v2.CenterCrop()),
        # "test": v2.Compose(),
    }

    # dataset.set_transform(data_transform)

    train_dataloader = DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        dataset["valid"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=16,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    progress = tqdm.tqdm(range(len(train_dataloader) * epoch))

    model = model.to("cuda")

    eval(model, test_dataloader, loss_fn)
    for _ in range(epoch):
        results_eval.clear()
        results_train.clear()

        loss_eval.clear()
        loss_train.clear()

        train(model, train_dataloader, loss_fn, optimizer, progress)

        eval(model, valid_dataloader, loss_fn)
        scheduler.step()

    print("Finshed Training")
    torch.save(model.state_dict(), "model.pt")
    # eval(model, test_dataloader, loss_fn)
