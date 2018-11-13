from utils import *
from models import AlexNet

import cv2
import torch
import numpy as np

from torch.autograd import Variable
import ipdb

mean_pts = 104.4724870017331
std_pts = 43.173022717543226
input_size = 227
extra = 60

face_cascade = cv2.CascadeClassifier(
    '/home/roach/.virtualenvs/nn/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml'
)
net = AlexNet()
net.load_state_dict(
    torch.load(
        'saved_models/keypoints_model.pt',
        map_location=lambda storage, loc: storage))
net.eval()

sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
original_sunglasses_width, original_sunglasses_height = sunglasses.shape[:2]

camera = cv2.VideoCapture(0)
while True:
    _, frame = camera.read()
    frame = cv2.flip(frame, 1)
    faces = face_cascade.detectMultiScale(frame, 1.25, 6)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

    for (x, y, w, h) in faces:
        face = frame[y - extra:y + h + extra, x - extra:x + w + extra]
        gray_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        gray_face = gray_face / 255.0

        gray_face = cv2.resize(
            gray_face, (input_size, input_size), interpolation=cv2.INTER_AREA)
        gray_face = gray_face.reshape(1, 1, input_size, input_size)
        gray_face = torch.from_numpy(gray_face)
        gray_face = gray_face.type(torch.FloatTensor)

        output_pts = net.forward(Variable(gray_face))
        output_pts = output_pts.view(68, 2).data.numpy()
        output_pts = output_pts * std_pts + mean_pts
        output_pts = output_pts * ((w + 2 * extra) / input_size,
                                   (h + 2 * extra) / input_size)

        angle = vec_ang(output_pts[27], output_pts[33])
        sunglasses_rotated = rotate_image(sunglasses, angle)
        original_sunglasses_rotated_width, orignal_sunglasses_rotated_height = sunglasses_rotated.shape[
            :2]

        sunglass_width = int(
            abs((output_pts[17][0] - output_pts[26][0]) * 1.1))
        sunglass_height = int(
            abs((output_pts[27][1] - output_pts[33][1]) / 1.1))
        sunglass_rotated_resized = cv2.resize(
            sunglasses_rotated, (int(sunglass_width * original_sunglasses_rotated_width / original_sunglasses_width),
                                 int(sunglass_height * orignal_sunglasses_rotated_height / original_sunglasses_height)),
            interpolation=cv2.INTER_CUBIC)
        transparent_region = sunglass_rotated_resized[:, :, :3] != 0

        # top-left location for sunglasses to go
        # 17 = edge of left eyebrow
        l_eyebrow_x = int(output_pts[17, 0] + x - extra - \
            (sunglass_rotated_resized.shape[1] - sunglass_width) / 2)
        l_eyebrow_y = int(output_pts[17, 1] + y - extra - \
            (sunglass_rotated_resized.shape[0] - sunglass_height) / 2)

        frame[l_eyebrow_y:l_eyebrow_y + sunglass_rotated_resized.shape[0],
              l_eyebrow_x:l_eyebrow_x + sunglass_rotated_resized.shape[1],
              :][transparent_region] = sunglass_rotated_resized[:, :, :3][transparent_region]

        cv2.imshow("Selfie Filters", frame)

        for pts in output_pts:
            cv2.circle(face, (int(pts[0]), int(pts[1])), 1, (0, 255, 0), 1)
        cv2.imshow("PTSs", face)
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # https://towardsdatascience.com/facial-keypoints-detection-deep-learning-737547f73515
