import numpy as np
import torch
from skimage.metrics import structural_similarity


def accuracy(output, target):
    with torch.no_grad():
        pred = torch.argmax(output, dim=1)
        assert pred.shape[0] == len(target)
        correct = 0
        correct += torch.sum(pred == target).item()
    return correct / len(target)


def top_k_acc(output, target, k=3):
    with torch.no_grad():
        pred = torch.topk(output, k, dim=1)[1]
        assert pred.shape[0] == len(target)
        correct = 0
        for i in range(k):
            correct += torch.sum(pred[:, i] == target).item()
    return correct / len(target)


def psnr(outputs, targets):
    value = 0
    for output, target in zip(outputs, targets):
        with torch.no_grad():
            original = target.cpu().detach().numpy()
            compared = output.cpu().detach().numpy()
            mse = np.mean(np.square(original - compared))
            value += np.clip(
                np.multiply(np.log10(255. * 255. / mse[mse > 0.]), 10.), 0., 99.99)[0]
    return value / len(outputs)


def ssim(outputs, targets):
    value = 0
    for output, target in zip(outputs, targets):
        with torch.no_grad():
            original = target.cpu().detach().numpy()
            compared = output.cpu().detach().numpy()
            original = np.moveaxis(original, 0, 2)
            compared = np.moveaxis(compared, 0, 2)
            value += structural_similarity(original, compared, multichannel=True)
    return value / len(outputs)
