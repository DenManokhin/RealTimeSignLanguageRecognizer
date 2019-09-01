import torch
import cv2 as cv
import numpy as np
from time import time
from pred import predict
from model import CNN


# load model parameters and copy it to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = torch.load("../weights.pth", map_location=device).to(device, torch.float)


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
        frame_blur = cv.GaussianBlur(frame_resized, (5, 5), 0)

        if calibrated:

            # select hand roi and apply mask by skin color
            hand_roi = frame_blur[y:y + h, x:x + w].copy()
            roi_ycrcb = cv.cvtColor(hand_roi, cv.COLOR_BGR2YCrCb)
            lower = np.array([60, max(0, cr - cr_diff), max(0, cb - cb_diff)], dtype="uint8")
            upper = np.array([255, min(255, cr + cr_diff), min(255, cb + cb_diff)], dtype="uint8")
            mask = cv.inRange(roi_ycrcb, lower, upper)

            # apply morphology operations to improve mask quality
            kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (7, 7))
            mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
            mask = cv.dilate(mask, kernel)

            # make prediction on gesture
            pred = predict(model, mask, device)
            char = class_map[pred.item()]
            cv.putText(frame_resized, "Char: " + char, (20, 430), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        else:

            # set color of central pixel of hand roi as skin color
            if cv.waitKey(1) & 0xFF == ord(" "):
                hand_roi = frame_blur[y:y + h, x:x + w].copy()
                roi_ycrcb = cv.cvtColor(hand_roi, cv.COLOR_BGR2YCrCb)
                _, cr, cb = roi_ycrcb[h // 2, w // 2, :]
                calibrated = True

            # mark hint where to place a hand
            cv.circle(frame_resized, (x + w // 2, y + h // 2), 3, (0, 255, 0), 3)
            mask = np.zeros((h, w))

        # fps counter
        end = time()
        fps = 1 / (end - start)
        cv.putText(frame_resized, "FPS: " + str(int(fps)), (20, 455), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # display frame
        cv.rectangle(frame_resized, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv.imshow("Gesture Recognizer", frame_resized)
        cv.imshow("Mask", mask)

        # exit if 'q' is pressed
        if cv.waitKey(1) & 0xFF == ord("q"):
            cv.destroyAllWindows()
            cap.release()
            break
    else:
        break
