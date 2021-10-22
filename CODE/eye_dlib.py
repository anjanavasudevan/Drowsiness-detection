import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
import matplotlib.pyplot as plt


image = cv2.imread("2058725932_1.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# gray = image.copy()


# Initialising the clssifiers:
# 1. Face
lbp_cascade = cv2.CascadeClassifier("lbpcascade_frontalface.xml")
# Detect face using dlib's facial detector
# detector = dlib.get_frontal_face_detector()
face = lbp_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5)
#face = detector(gray, 0)

# 2. eye detection
# Get shape using dlib's shape predictor module
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Get eye landmarks from both left and right eyes respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# Eye aspect ratio function


def eye_aspect_ratio(eye_features):
    A = dist.euclidean(eye_features[0], eye_features[3])
    B = dist.euclidean(eye_features[1], eye_features[4])
    C = dist.euclidean(eye_features[2], eye_features[5])

    aspect_ratio = (A+B) / (2.0*C)

    return aspect_ratio


# Loop over the faces:
for (x, y, w, h) in face:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 5)
    dlib_rect = dlib.rectangle(int(x), int(y), int(x+w), int(y+h))

for faces in face:
    # where dlib_rect contains info on the no. of faces
    shape1 = predict(gray, dlib_rect)
    shape = face_utils.shape_to_np(shape1)

    # Extract regions of the eyes:
    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    left_eye = shape[lStart:lEnd]
    right_eye = shape[rStart:rEnd]

    # Calculating eye aspect ratio
    EAR_left = eye_aspect_ratio(left_eye)
    EAR_right = eye_aspect_ratio(right_eye)
    EAR = (EAR_left+EAR_right)/2
    print('Eye aspect ratio =  0.2f', EAR)

    if EAR < 0.5:
        print("Drowsy")
    else:
        print("not drowsy")

    leftEyeHull = cv2.convexHull(left_eye)
    rightEyeHull = cv2.convexHull(right_eye)

    # Drawing the contours -ve no. all the contours
    cv2.drawContours(image, [leftEyeHull], -1, (0, 255, 0), 2)
    cv2.drawContours(image, [rightEyeHull], -1, (0, 255, 0), 2)

    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.show()
    # Landmarks
