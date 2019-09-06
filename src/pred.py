import torch
import numpy as np
import cv2 as cv
from model import CNN


def predict(model:CNN, image:np.ndarray, device:torch.device):
    image_resized = cv.resize(image, (28, 28))
    image_tensor = torch.from_numpy(image_resized).to(device, torch.float)
    tensor_norm = image_tensor / 255
    input = tensor_norm.view((1, 1, 28, 28))

    with torch.no_grad():
        model.eval()
        model.to(device, torch.float)
        outputs = model(input)
        _, pred = outputs.max(1)

    return pred
