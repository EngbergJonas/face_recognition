import cv2 as cv
import dlib
import numpy as np
import os
from imutils import face_utils
import argparse
from scipy.spatial import distance as dist

subjects = ["", "Jonas"]
p = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)


# This function parses the directory of pictures and gathers training data
# to be used with the LBPH face_recognizer
def prepare_training_data(data_folder_path):

    dirs = os.listdir(data_folder_path)

    faces = []
    labels = []

    for dir_name in dirs:

        if not dir_name.startswith("s"):
            continue

        label = int(dir_name.replace("s", ""))
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)

        for image_name in subject_images_names:

            if image_name.startswith("."):
                continue

            image_path = subject_dir_path + "/" + image_name
            image = cv.imread(image_path)

            cv.imshow("Training on image...", cv.resize(image, (400, 500)))
            cv.waitKey(100)
            face, rect = detect_face(image)

            if face is not None:
                faces.append(face)
                labels.append(label)

    cv.destroyAllWindows()
    cv.waitKey(1)
    cv.destroyAllWindows()

    return faces, labels


# This function is solibly used with to detect the faces in the pictures used
# to when training the face detector
def detect_face(img):
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    face_cascade = cv.CascadeClassifier(
        'opencv-files/haarcascade_frontalface_default.xml')

    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5)

    if (len(faces) == 0):
        return None, None

    (x, y, w, h) = faces[0]

    return gray[y:y+w, x:x+h], faces[0]


# This function takes the total time of frames / total time of eyes being closed
def eye_close_ratio(blink, frames):
    result = blink / frames * 100
    return round(result)


# This function counts the aspect ratio of the eyes, and determines how open they
# are
def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    # computing the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    return ear


faces, labels = prepare_training_data("training-data")

print ("Total faces: ", len(faces))
print ("total labels:", len(labels))

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# use and train the face_recognizer with the trained data
face_recognizer = cv.face.LBPHFaceRecognizer_create()
face_recognizer.train(faces, np.array(labels))

# Define the variables to be used
counter = 0
ear_threshold = 0.26
ear_alarm = 20
blink = 0.0
frames = 0.0

# Start the video capture
capture = cv.VideoCapture(0)
while True:
    # define the frame from the video
    rect, frame = capture.read()
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    rects = detector(gray, 0)

    # make a cascade face for detection
    face_cascade = cv.CascadeClassifier(
        'opencv-files/haarcascade_frontalface_default.xml')
    face = face_cascade.detectMultiScale(
        gray, scaleFactor=1.05, minNeighbors=4, minSize=(30, 30))

    # draw a rectangle with the label that it closest resembles to, based on the data trained
    for(x, y, w, h) in face:
        cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y: y + h, x: x + w]

        label = face_recognizer.predict(roi_gray)
        label_text = subjects[label[0]]

        cv.putText(frame, label_text, (x, y - 5),
                   cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    # find the eyes in the detected face and draw the countours of the eyes
    # warn the user if he/she is getting tired based on the PERCLOSE (percentage of eyes being closed)
    # and determine if the user is sleeping/too tired to drive
    for rect in rects:
        frames += 1
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart: rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        EAR = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv.convexHull(leftEye)
        rightEyeHull = cv.convexHull(rightEye)

        cv.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if EAR < ear_threshold:
            counter += 1
            blink += 1
            PERCLOS = eye_close_ratio(blink, frames)
            print(PERCLOS, "%")

            if PERCLOS <= 30.0:
                cv.putText(frame, "You are starting to drowse, EAR: {:.2f}".format(EAR), (300, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if PERCLOS >= 30.0:
                cv.putText(frame, "You are too tired to drive!", (300, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            if counter >= ear_alarm:
                cv.putText(frame, "Wake up!", (10, 30),
                           cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            counter = 0
            cv.putText(frame, "Eyes are open, EAR: {:.2f}".format(EAR), (300, 30),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv.imshow(subjects[0], frame)
    key = cv.waitKey(1)
    if key == ord("q"):
        False
        break

capture.release()
cv.destroyAllWindows()
