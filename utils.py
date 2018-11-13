import math

import cv2


def vec_ang(vec1, vec2):
    """
    vec1, pts28
    vec2, pts34
    """
    angle = math.atan2(vec2[0]-vec1[0], vec2[1] - vec1[1])
    if vec1[1] < vec2[1]:
        angle = math.degrees(angle)
    else:
        angle = math.degrees(angle) + 180
    return angle


def rotate_image(mat, angle):
    height, width = mat.shape[:2]
    image_center = (width / 2, height / 2)

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1)

    radians = math.radians(angle)
    sin = math.sin(radians)
    cos = math.cos(radians)
    bound_w = int((height * abs(sin)) + (width * abs(cos)))
    bound_h = int((height * abs(cos)) + (width * abs(sin)))

    rotation_mat[0, 2] += ((bound_w / 2) - image_center[0])
    rotation_mat[1, 2] += ((bound_h / 2) - image_center[1])

    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat
