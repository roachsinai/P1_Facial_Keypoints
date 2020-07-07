import cv2
import dlib
import torch

from imutils import face_utils
from utils import *

face_landmark_path = '/home/roach/.dlib/shape_predictor_68_face_landmarks.dat'

sunglasses = cv2.imread('images/sunglasses.png', cv2.IMREAD_UNCHANGED)
original_sunglasses_height, original_sunglasses_width= sunglasses.shape[:2]

camera = cv2.VideoCapture(0)
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(face_landmark_path)

while True:
    ret , frame = camera.read()
    frame = cv2.flip(frame, 1)
    if ret:
        face_rects = detector(frame, 0)
        if len(face_rects) == 0:
            continue

    for face_rect in face_rects:
        output_pts = predictor(frame, face_rect)
        output_pts = face_utils.shape_to_np(output_pts)

        (x, y, w, h) = face_utils.rect_to_bb(face_rect)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)

        face = frame[y:y + h, x:x + w]

        angle = vec_ang(output_pts[27], output_pts[33])
        sunglasses_rotated = rotate_image(sunglasses, angle)
        original_sunglasses_rotated_height, original_sunglasses_rotated_width= sunglasses_rotated.shape[
            :2]

        sunglass_width = int(
            abs((output_pts[17][0] - output_pts[26][0]) * 1.1))
        sunglass_height = int(
            abs((output_pts[27][1] - output_pts[33][1]) / 1.1))
        sunglass_rotated_resized = cv2.resize(
            sunglasses_rotated, (int(original_sunglasses_rotated_width * (sunglass_width / original_sunglasses_width)),
                                 int(original_sunglasses_rotated_height * (sunglass_height / original_sunglasses_height))),
            interpolation=cv2.INTER_CUBIC)
        transparent_region = sunglass_rotated_resized[:, :, :3] != 0

        # top-left location for sunglasses to go
        # 17 = edge of left eyebrow
        eye_center_x = int((output_pts[39, 0] + output_pts[42, 0]) / 2)
        eye_center_y = int((output_pts[39, 1] + output_pts[42, 1]) / 2)
        sunglass_rotated_resized_top_pos = eye_center_y - sunglass_rotated_resized.shape[0] // 2
        sunglass_rotated_resized_left_pos = eye_center_x - sunglass_rotated_resized.shape[1] // 2

        frame[sunglass_rotated_resized_top_pos:sunglass_rotated_resized_top_pos + sunglass_rotated_resized.shape[0],
              sunglass_rotated_resized_left_pos:sunglass_rotated_resized_left_pos + sunglass_rotated_resized.shape[1],
              :][transparent_region] = sunglass_rotated_resized[:, :, :3][transparent_region]

        cv2.imshow("Selfie Filters", frame)

        for pts in output_pts:
            cv2.circle(face, (int(pts[0] - x), int(pts[1] - y)), 1, (0, 255, 0), 1)
        cv2.imshow("PTSs", face)
    # If the 'q' key is pressed, stop the loop
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

    # https://towardsdatascience.com/facial-keypoints-detection-deep-learning-737547f73515
