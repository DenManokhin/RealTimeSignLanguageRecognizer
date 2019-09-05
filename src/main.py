import torch
import cv2 as cv
import numpy as np
from time import time
from pred import predict
from model import CNN


# load model parameters and copy it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../model_mnist_augmented.pth", map_location=device).to(device, torch.float)


calibrated = False
window_size = (640, 480)
# initial hand roi coordinates
x, y, w, h = (50, 50, 250, 250)
# skin color cb and cr components delta
cb_diff, cr_diff = (10, 10)
# CNN output -> American Sign Language
class_map = {
    0 : "A",
    1 : "B",
    2 : "C",
    3 : "G",
    4 : "L",
    5 : "V",
    6 : "W",
    7 : "Y"
}


cap = cv.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    # initialize timer for fps counter
    start = time()
    if ret:

        # preprocess image
        frame_resized = cv.resize(frame, window_size)
        # frame_blur = cv.GaussianBlur(frame_resized, (5, 5), 0)

        if calibrated:

            # select hand roi and convert to grayscale
            hand_roi = frame_resized[y:y + h, x:x + w].copy()
            hand_roi_gray = cv.cvtColor(hand_roi, cv.COLOR_BGR2GRAY)

            # make prediction on gesture
            pred = predict(model, hand_roi_gray, device)
            # char = class_map[pred.item()]
            cv.putText(frame_resized, "Char: " + str(pred), (20, 430), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        else:

            if cv.waitKey(1) & 0xFF == ord(" "):
                calibrated = True

        # fps counter
        end = time()
        fps = 1 / (end - start)
        cv.putText(frame_resized, "FPS: " + str(int(fps)), (20, 455), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # display frame
        cv.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow("Gesture Recognizer", frame_resized)

        # exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            cap.release()
            break
    else:
        break
