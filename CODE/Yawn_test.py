import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt

FACE_DETECTOR = "lbpcascade_frontalface.xml"
SHAPE_PREDICT = "shape_predictor_68_face_landmarks.dat"
lbp_cascade = cv2.CascadeClassifier(FACE_DETECTOR)
predictor = dlib.shape_predictor(SHAPE_PREDICT)


def get_landmarks(im):
    rects = lbp_cascade.detectMultiScale(im, scaleFactor=1.2, minNeighbors=5)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"

    for (x, y, w, h) in rects:
        cv2.rectangle(im, (x, y), (x+w, y+h), (0, 255, 0), 5)
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

    return np.matrix([[p.x, p.y] for p in predictor(im, dlib_rect).parts()])


def annotate_landmarks(im, landmarks):
    im = im.copy()
    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))
    return im


def top_lip(landmarks):
    top_lip_pts = []
    for i in range(50, 53):
        top_lip_pts.append(landmarks[i])
    for i in range(61, 64):
        top_lip_pts.append(landmarks[i])
    top_lip_all_pts = np.squeeze(np.asarray(top_lip_pts))
    top_lip_mean = np.mean(top_lip_pts, axis=0)
    return int(top_lip_mean[:, 1])


def bottom_lip(landmarks):
    bottom_lip_pts = []
    for i in range(65, 68):
        bottom_lip_pts.append(landmarks[i])
    for i in range(56, 59):
        bottom_lip_pts.append(landmarks[i])
    bottom_lip_all_pts = np.squeeze(np.asarray(bottom_lip_pts))
    bottom_lip_mean = np.mean(bottom_lip_pts, axis=0)
    return int(bottom_lip_mean[:, 1])


def yawn(im):
    landmarks_np = get_landmarks(image)
    if landmarks_np == "error":
        return im, 0

    lip = landmarks_np[48:68]
    print(lip)
    im = annotate_landmarks(im, lip)

    upper_lip = top_lip(landmarks_np)
    lower_lip = bottom_lip(landmarks_np)

    lip_distance = abs(upper_lip - lower_lip)

    return im, lip_distance


image = cv2.imread("2058725932_1.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

im, lip_distance = yawn(gray)

if lip_distance > 25:
    print("You are yawning")

plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
plt.show()
