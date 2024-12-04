"""
This code is based on code examples given in
[Elgendy M (2020) Deep Learning for Vision Systems. Manning Publications ].

Originally Keras was used for them and were adapted to pytorch.
"""

import os
import PIL.Image
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
import timm

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


def eval(model, dataloader: DataLoader, loss_fn, device="cuda", epoch=0):
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

    print(
        f"\nEpoch {epoch} = Test Accuracy: {(100*correct):>0.1f}%, Test Avg loss: {test_loss:>8f} \n"
    )


def collate_fn(batch):
    result = dict()
    result["target"] = torch.tensor([x["target"] for x in batch], dtype=torch.float32)
    result["data"] = torch.tensor([x["data"] for x in batch], dtype=torch.float32)

    return result


idx_to_label = {
    0: "cat",
    1: "dog",
}

if __name__ == "__main__":
    epoch = 20

    input_channels = 3
    base_hidden_units = 32

    model = timm.create_model("vgg16.tv_in1k", pretrained=True)

    # Freeze Network
    model.features.requires_grad_(False)
    # model.pre_logits.requires_grad_(False)

    # Create new head
    model.head = timm.layers.classifier.ClassifierHead(model.head.in_features, 2)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    scheduler = LinearLR(optimizer, total_iters=epoch)

    summary(model, (3, 224, 224), device="cpu")

    # Prepare data
    dataset_raw = datasets.load_dataset("TransferLearning/01_data")

    print(dataset_raw)

    print(model)

    unique_labels = len(np.unique(dataset_raw["train"]["label"]))
    dataset_onehot = dataset_raw.map(
        lambda x: {
            "target": [0 if v != x["label"] else 1 for v in range(unique_labels)]
        }
    )

    dataset_same_size = dataset_onehot.map(
        lambda x: {"img": x["image"].resize((224, 224))}
    )

    # Convert Image
    # The base images are ordered in Height, Width and Color, but
    # the Model expects it to be Color, Height, Width,
    # So the images are here transposed to seperate the color channels into their own
    # images
    dataset_converted_image = dataset_same_size.map(
        lambda x: {
            "data": torch.from_numpy(
                np.array(x["img"]).astype(np.float32),
            )
            .transpose(0, -1)
            .transpose(1, 2)
        }
    )

    dataset = dataset_converted_image.remove_columns("image")
    dataset = dataset_converted_image.remove_columns("img")
    dataset = dataset.remove_columns("label")

    print(dataset)

    # Some data augmentation
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
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    valid_dataloader = DataLoader(
        dataset["validation"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    test_dataloader = DataLoader(
        dataset["test"],
        collate_fn=collate_fn,
        num_workers=4,
        batch_size=4,
        shuffle=True,
        pin_memory=True,
        prefetch_factor=100,
        persistent_workers=True,
    )

    progress = tqdm.tqdm(range(len(train_dataloader) * epoch))

    model = model.to("cuda")

    eval(model, test_dataloader, loss_fn)
    for e in range(epoch):
        results_eval.clear()
        results_train.clear()

        loss_eval.clear()
        loss_train.clear()

        train(model, train_dataloader, loss_fn, optimizer, progress)

        eval(model, valid_dataloader, loss_fn, epoch=e)
        scheduler.step()

    print("Finshed Training")
    torch.save(model.state_dict(), "model_01.pt")
    eval(model, test_dataloader, loss_fn)
