import numpy as np
import cv2
import dlib
import imutils
from imutils import face_utils
from scipy.spatial import distance as dist
# from playsound import playsound


FACE_DETECTOR = "lbpcascade_frontalface.xml"
SHAPE_PREDICT = "shape_predictor_68_face_landmarks.dat"
lbp_cascade = cv2.CascadeClassifier(FACE_DETECTOR)
predictor = dlib.shape_predictor(SHAPE_PREDICT)


def get_landmarks(im):
    # Detect faces
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


# Test in Webcam
cap = cv2.VideoCapture(0)
eye_closed = 0
yawns = 0
eye_status = False

while(True):
    # Capture frame-by-frame
    prev_eye_status = eye_status
    ret, frame = cap.read()
    im1, EAR = eye_drowsy(frame)
    image, lip_dist = yawn(frame)

    # Test drowsiness
    if EAR <= 0.25:
        eye_status = True
    else:
        eye_status = False

    if prev_eye_status == True and eye_status == True:
        eye_closed += 1

    if lip_dist >= 30:
        yawns += 1
        cv2.putText(image, "You are yawning", (50, 450),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)

    if eye_closed >= 40 or yawns >= 5:
        cv2.putText(image, "You are drowsy, quit driving", (50, 50),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 2)
        # playsound("Alarm.mp3")

    # Display the resulting frame
    cv2.imshow('frame', frame)
    cv2.imshow('Landmarks', image)
    if cv2.waitKey(1) == 13:
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
