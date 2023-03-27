# https://stackoverflow.com/questions/49397489/using-getperspective-in-opencv
import cv2
import numpy as np

img = cv2.imread("input-getperspective.jpg")

src = np.array([[0, 0], [997, 102], [1000, 600], [0, 995]], np.float32)
dst = np.array([[0, 0], [997, 0], [1000, 995], [0, 995]], np.float32)

M = cv2.getPerspectiveTransform(src, dst)
warped = cv2.warpPerspective(img, M, (1000, 1000))

cv2.imwrite("output.jpg", warped)

