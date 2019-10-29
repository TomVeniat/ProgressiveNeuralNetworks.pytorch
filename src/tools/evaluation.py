import torch
from tqdm import tqdm


def evaluate_model(model, x, y, dataset_loader, **kwargs):
    total = 0
    correct = 0
    for images, labels in tqdm(dataset_loader, ascii=True):
        x.resize_(images.size()).copy_(images)
        y.resize_(labels.size()).copy_(labels)

        with torch.no_grad():
            preds = model(x.view(x.size(0), -1), **kwargs)

        _, predicted = torch.max(preds.data, 1)

        total += labels.size(0)
        correct += (predicted == y).sum().item()

    return 100 * correct / total