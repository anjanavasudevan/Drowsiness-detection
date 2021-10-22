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
    rects = lbp_cascade.detectMultiScale(im, scaleFactor=1.1, minNeighbors=5)

    if len(rects) > 1:
        return "error"
    if len(rects) == 0:
        return "error"

    for (x, y, w, h) in rects:
        dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

    return np.matrix([[p.x, p.y] for p in predictor(im, dlib_rect).parts()])

# Extract eye regions:


def eye_region(im, landmarks):

    left_eye = []
    left_eye = landmarks[36:42]
    im = annotate(im, left_eye)

    right_eye = []
    right_eye = landmarks[42:48]
    im = annotate(im, right_eye)

    EAR_left = eye_aspect_ratio(left_eye)
    EAR_right = eye_aspect_ratio(right_eye)
    EAR = (EAR_left + EAR_right) / 2
    return im, EAR


def annotate(im, landmarks):

    for idx, point in enumerate(landmarks):
        pos = (point[0, 0], point[0, 1])
        cv2.putText(im, str(idx), pos,
                    fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                    fontScale=0.4,
                    color=(0, 0, 255))
        cv2.circle(im, pos, 3, color=(0, 255, 255))

    return im


def eye_aspect_ratio(parts):
    A = dist.euclidean(parts[0, 1], parts[3, 1])
    B = dist.euclidean(parts[1, 1], parts[4, 1])
    C = dist.euclidean(parts[2, 1], parts[5, 1])
    aspect_ratio = (A+B) / (2.0*C)
    return aspect_ratio


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


def yawning(im, landmarks):
    lip = landmarks[48:68]
    im = annotate(im, lip)

    upper_lip = top_lip(landmarks)
    lower_lip = bottom_lip(landmarks)
    lip_distance = abs(upper_lip - lower_lip)

    return im, lip_distance


def eye_drowsy(im):
    landmarks_np = get_landmarks(im)

    if landmarks_np == "error":
        return im, 0

    im_landmarks, EAR = eye_region(im, landmarks_np)
    return im_landmarks, EAR


def yawn(im):
    landmarks_np = get_landmarks(im)

    if landmarks_np == "error":
        return im, 0
    im_landmarks, lip_dist = yawning(im, landmarks_np)

    return im_landmarks, lip_dist


image = cv2.imread("100466187_1.jpg")
# gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

final, EAR = eye_drowsy(gray)
final, lip_dist = yawn(gray)

if lip_dist > 25:
    print("You are yawning")

if EAR <= 0.5 or lip_dist > 25:
    print("You are Drowsy")

print(lip_dist)
plt.imshow(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
plt.show()
