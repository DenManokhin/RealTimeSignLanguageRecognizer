import torch


def test(model, dataloader, device):
    with torch.no_grad():
        total = 0
        correct = 0
        model.eval()
        model.to(device, torch.float)
        for data in dataloader:
            inputs, labels = [data[0].to(device), data[1].to(device)]
            inputs = inputs.float()
            outputs = model(inputs)
            _, pred = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (pred == labels).sum().item()
        accuracy = (correct / total) * 100
        return accuracy
