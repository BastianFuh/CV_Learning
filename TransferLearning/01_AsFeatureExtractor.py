"""
This code is based on code examples given in
[Elgendy M (2020) Deep Learning for Vision Systems. Manning Publications ].

Originally Keras was used for them and were adapted to pytorch.
"""

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LinearLR
import torch.nn.functional as F
import torch.nn as nn
import datasets
from torchsummary import summary
import tqdm
from torchvision.transforms import v2 as transforms
from torchvision.models import vgg16

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
        data = batch["image"].to(device)
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


def eval(
    model, dataloader: DataLoader, loss_fn, device="cuda", epoch=0, extra_print=""
):
    model.eval()

    dataset_size = len(dataloader.dataset)
    num_batches = len(dataloader)

    test_loss, correct = 0.0, 0.0

    progress_bar = tqdm.tqdm(range(num_batches))

    with torch.no_grad():
        for batch in dataloader:
            data = batch["image"].to(device)
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

    progress_bar.write(
        f"{extra_print} - Epoch {epoch} = Test Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f}\n"
    )

    return (correct, test_loss)


def collate_fn(batch):
    result = dict()
    result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["image"] = torch.stack([x["image"] for x in batch])

    return result


def augment_dataset(dataset):
    pass


idx_to_label = {
    0: "cat",
    1: "dog",
}

if __name__ == "__main__":
    epoch = 50

    input_channels = 3
    base_hidden_units = 64

    model = vgg16()

    print(model)

    # Freeze Network
    model.features.requires_grad_(False)

    # Create new head
    hidden_layer_size = 128
    model.classifier = nn.Sequential(
        nn.Linear(in_features=25088, out_features=hidden_layer_size),
        nn.ReLU(),
        nn.BatchNorm1d(int(hidden_layer_size)),
        nn.Dropout(0.5),
        nn.Linear(in_features=int(hidden_layer_size), out_features=2),
    )

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    summary(model, (3, 224, 224), device="cpu")

    # Prepare data
    dataset_raw = datasets.load_dataset("TransferLearning/01_data")

    dataset_raw["train"]

    print(dataset_raw)

    print(model)

    unique_labels = len(np.unique(dataset_raw["train"]["label"]))
    dataset_onehot = dataset_raw.map(
        lambda x: {
            "target": [0 if v != x["label"] else 1 for v in range(unique_labels)]
        }
    )

    dataset = dataset_onehot.remove_columns("label")
    # dataset = dataset.remove_columns("image")

    print(dataset)

    # Some data augmentation
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.Resize((250, 250)),
                # transforms.FiveCrop((224, 224)),
                transforms.RandomHorizontalFlip(0.5),
                transforms.RandomCrop(224),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "valid": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
        "test": transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToImage(),
                transforms.ToDtype(torch.float32, scale=True),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        ),
    }

    dataset["train"].set_transform(data_transform["test"])
    dataset["validation"].set_transform(data_transform["valid"])
    dataset["test"].set_transform(data_transform["train"])

    train_dataloader = DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=30,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        dataset["validation"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=30,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        dataset["train"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=50,
        shuffle=False,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    progress = tqdm.tqdm(range(len(train_dataloader) * epoch))

    model = model.to("cuda")

    eval(model, valid_dataloader, loss_fn, extra_print="Initial Results")
    for e in range(epoch):
        train(model, train_dataloader, loss_fn, optimizer, progress)

        precision, loss = eval(
            model, train_dataloader, loss_fn, epoch=e, extra_print="Train Eval"
        )
        results_train.append(precision)
        loss_train.append(loss)

        precision, loss = eval(
            model, valid_dataloader, loss_fn, epoch=e, extra_print="Validate Eval"
        )
        results_eval.append(precision)
        loss_eval.append(loss)

    progress.write("Finshed Training")
    torch.save(model.state_dict(), "model_01.pt")
    eval(model, test_dataloader, loss_fn)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Train")
    ax1.set_title("Precision")
    ax1.plot(range(epoch), results_train)
    ax2.set_title("Loss")
    ax2.plot(range(epoch), loss_train)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.suptitle("Eval")
    ax1.set_title("Precision")
    ax1.plot(range(epoch), results_eval)
    ax2.set_title("Loss")
    ax2.plot(range(epoch), loss_eval)
    plt.show()
