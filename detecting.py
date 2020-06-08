# import the packages
import numpy as np
import time
from collections import Counter
import imutils
import dlib
import cv2
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from cnn_model import focal_loss
from drowsy_landmark import sound_alarm, eye_aspect_ratio, mouth_aspect_ratio, circularity
from argument_parser import arguments
import threading
import recommend
import result_toweb


# argparse arguments function call
args = arguments()

# driver Status Analysis Collection Variables
e_state = []
d_state = []
state_rec = None

# initialize the frame counter as well as a boolean used to indicate if the alarm is going off
EAR_LIST = []
NOR_EAR = 0
COUNTER = 0
ALARM_ON = False

# grab the indexes of the facial landmarks for the left and right eye, mouth respectively
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
(mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]

# drowsy face detector and the facial landmark predictor create
dro_detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
# emotion face detector create and load model, emotion labels
emo_detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
model = load_model(args['model'], custom_objects={'focal_loss': focal_loss})
EMOTIONS = ['angry', 'happy', 'neutral']

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing the stream)
outputFrame = None
outputProb = None
lock = threading.Lock()

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
# start the video stream thread
vs = VideoStream(src=args["webcam"]).start()
time.sleep(1.0)
print("[INFO] loading facial landmark predictor...")
print("[INFO] starting video stream thread...")

song_csv = "C:/Users/YEO/PycharmProjects/Final_Project/arena_data/song.csv"
train_json = 'C:/Users/YEO/PycharmProjects/Final_Project/arena_data/data.json'
my_id = 126993


def driver_detect():
    # grab global references to the video stream, output frame, and
    # lock variables
    global vs, outputFrame, outputProb, lock, NOR_EAR, COUNTER, ALARM_ON, state_rec

    state_rec = None
    EAR_LIST = []
    # loop over frames from the video stream

    while state_rec is None:
        # grab the frame from the threaded video file stream, resize
        # it, and convert it to grayscale
        # channels
        frame = vs.read()
        frame = imutils.resize(frame, width=450)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # detect faces in the grayscale frame
        canvas = np.zeros((120, 450, 3), dtype='uint8')
        dro_rects = dro_detector(gray, 0)
        emo_rects = emo_detector.detectMultiScale(gray, scaleFactor=1.1,
                                                  minNeighbors=5, minSize=(30, 30),
                                                  flags=cv2.CASCADE_SCALE_IMAGE)

        # emotions detections
        if len(emo_rects) > 0:
            # face area
            rect = sorted(emo_rects, reverse=True, key=lambda x: (x[2] - x[0]) * (x[3] - x[1]))[0]
            (fX, fY, fW, fH) = rect

            roi = gray[fY:fY + fH, fX:fX + fW]
            roi = cv2.resize(roi, (48, 48))
            roi = roi.astype('float') / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            # preds
            preds = model.predict(roi)[0]
            label = EMOTIONS[preds.argmax()]
            e_state.append(label)

            for (i, (emotion, prob)) in enumerate(zip(EMOTIONS, preds)):
                text = "{}: {:.2f}%".format(emotion, prob * 100)

                w = int(prob * 300)
                cv2.rectangle(canvas, (5, (i * 35) + 5),
                              (w, (i * 35) + 35), (0, 0, 225), -1)
                cv2.putText(canvas, text, (10, (i * 35) + 23),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 2)

                cv2.putText(frame, label, (fX, fY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 0, 0), 2)
                cv2.rectangle(frame, (fX, fY), (fX + fW, fY + fH), (255, 0, 0), 2)

        # loop over the drowsy face detections
        for dro in dro_rects:
            # determine the facial landmarks for the face region, then
            # convert the facial landmark (x, y)-coordinates to a NumPy
            # array
            shape = predictor(gray, dro)
            shape = face_utils.shape_to_np(shape)

            # extract the left and right eye coordinates, then use the
            # coordinates to compute the eye aspect ratio for both eyes
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = eye_aspect_ratio(leftEye)
            rightEAR = eye_aspect_ratio(rightEye)

            inMouth = shape[mStart:mEnd]
            inMAR = mouth_aspect_ratio(inMouth)

            eyes = (leftEye + rightEye)
            ear = (leftEAR + rightEAR) / 2.0

            EAR_LIST.append(ear)
            if len(EAR_LIST) == 30:
                NOR_EAR = sum(EAR_LIST) / 30

            mar = (inMAR)
            cir = circularity(eyes)
            moe = mar / ear

            # compute the convex hull for the left and right eye, then
            # visualize each of the eyes
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

            inMouthHull = cv2.convexHull(inMouth)
            cv2.drawContours(frame, [inMouthHull], -1, (0, 255, 0), 1)

            # check to see if the eye aspect ratio is below the blink
            # threshold, and if so, increment the blink frame counter
            if ear < NOR_EAR - args["EYE_AR_THRESH"] or mar > args["MOUTH_AR_THRESH"] or cir < args[
                "CIR_THRESH"] or moe > args["MOE_THRESH"]:
                COUNTER += 1

                # if the eyes were closed for a sufficient number of
                # then sound the alarm
                if COUNTER >= args["CONSEC_FRAMES"] or (COUNTER / args["CONSEC_FRAMES2"]) * 100 >= 37:
                    # if the alarm is not on, turn it on
                    if not ALARM_ON:
                        ALARM_ON = True
                        # check to see if an alarm file was supplied,
                        # and if so, start a thread to have the alarm
                        # sound played in the background
                        if args["alarm"] != "":
                            s = Thread(target=sound_alarm,
                                       args=(args["alarm"],))
                            s.deamon = True
                            s.start()

                    # draw an alarm on the frame
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if ALARM_ON == True and state_rec is None:
                        state_rec = 'drowsy'
                        print(state_rec)
                        del e_state[:]
                        del d_state[:]
                        COUNTER = 0

                        ply_title, ply_tag = recommend.main(song_csv, train_json, state_rec, my_id)
                        result_toweb.autoplay(ply_tag, ply_title)

                ALARM_ON = False



            else:
                COUNTER = 0
                ALARM_ON = False
                d_state.append('awake')

            # draw the computed aspect ratio on the frame to help
            # with debugging and setting the correct eye aspect ratio
            # thresholds and frame counters
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "CIR: {:.2f}".format(cir), (300, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "MAR: {:.2f}".format(mar), (300, 70),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "MOE: {:.2f}".format(moe), (300, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

            cv2.putText(frame, "N_EAR: {:.2f}".format(NOR_EAR), (300, 110),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # extract driver status analysis results

        if len(e_state) == 220 and state_rec == None:
            # if Counter(d_state).most_common(1)[0][0] == 'awake':
            state_rec = Counter(e_state).most_common(1)[0][0]
            del e_state[:]
            del d_state[:]
            ply_title, ply_tag = recommend.main(song_csv, train_json, state_rec, my_id)
            print(state_rec)
            result_toweb.autoplay(ply_tag, ply_title)
            # else:
            # state_rec = Counter(d_state).most_common(1)[0][0]
            # del e_state[:]
            # del d_state[:]
            # ply_title, ply_tag = recommend.main(song_csv, train_json, state_rec, my_id)
            # # print(state_rec)
            # result_toweb.autoplay(ply_tag, ply_title)

        # acquire the lock, set the output frame, and release the
        # lock

        with lock:
            outputFrame = frame.copy()
            outputProb = canvas.copy()

    return state_rec


def f_generate():
    # grab global references to the output frame and lock variables
    global outputFrame, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputFrame is None:
                continue
            # encode the frame in JPEG format
            (flag, encodedImage) = cv2.imencode(".jpg", outputFrame)
            # ensure the frame was successfully encoded
            if not flag:
                continue
        # yield the output frame in the byte format
        yield (b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage) + b'\r\n')


def p_generate():
    # grab global references to the output frame and lock variables
    global outputProb, lock
    # loop over frames from the output stream
    while True:
        # wait until the lock is acquired
        with lock:
            # check if the output frame is available, otherwise skip
            # the iteration of the loop
            if outputProb is None:
                continue
            # encode the frame in JPEG format
            (flag2, encodedImage2) = cv2.imencode(".jpg", outputProb)
            # ensure the frame was successfully encoded
            if not flag2:
                continue
        # yield the output frame in the byte format
        yield (b'--canvas\r\n' b'Content-Type: image/jpeg\r\n\r\n' +
               bytearray(encodedImage2) + b'\r\n')