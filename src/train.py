from time import time
from test import test


def train(model, dataloader, criterion, optimizer, epochs_count, device, scheduler=None, testloader=None):

    losses = []
    accuracies = []

    model.train()
    model.to(device)

    start = time()

    for epoch in range(epochs_count):
        running_loss = 0.0

        for i, data in enumerate(dataloader):
            progress = (i * dataloader.batch_size / len(dataloader.dataset)) * 100
            print("Epoch {}. Training... {:.2f}%".format(epoch, progress), end="\r", flush=True)

            inputs, labels = [data[0].to(device), data[1].to(device)]
            inputs = inputs.float()
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            if scheduler:
                scheduler.step()
            running_loss += loss.item()

        running_loss /= len(dataloader.dataset)
        losses.append(running_loss)

        print("Epoch No: {} | Loss: {:.7f}".format(
            epoch, running_loss), end=" ")

        if testloader:
            test_acc = test(model, testloader, device)
            accuracies.append(test_acc)
            print("| Accuracy: {:.2f}%".format(test_acc))
        else:
            print("")

    end = time()
    spent = int(end - start)
    minutes, seconds = (spent // 60, spent % 60)
    print("Finished training. Elapsed time: {} m {} s".format(minutes, seconds))
    return losses, accuracies


if __name__ == "__main__":

    import argparse
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from model import CNN
    from dataset import GesturesDataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--batch", type=int, help="Batch size used for training", default=256)
    parser.add_argument("--workers", type=int, help="Number of workers used for loading data", default=0)
    parser.add_argument("--epochs", type=int, help="Training epochs count", default=50)
    parser.add_argument("--out", type=str, help="Output name", default="model.pth")
    args = vars(parser.parse_args())

    batch_size = args["batch"]
    num_workers = args["workers"]
    epochs = args["epochs"]
    out = args["out"]

    trainset = GesturesDataset("../data/train.csv")
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    testset = GesturesDataset("../data/test.csv")
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNN().to(device, torch.float)

    optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9)

    criterion = nn.CrossEntropyLoss()

    losses, accuracies = train(model, trainloader, criterion, optimizer, epochs, device, testloader=testloader)
    torch.save(model, out)
