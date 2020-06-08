from scipy.spatial import distance as dist
import playsound
import math

def sound_alarm(path):
    # play an alarm sound
    playsound.playsound(path)

def eye_aspect_ratio(eye):
    # compute the euclidean distances between the two sets of
    # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    # compute the euclidean distance between the horizontal
    # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[0], eye[3])

    # compute the eye aspect ratio
    ear = (A + B) / (2.0 * C)

    # return the eye aspect ratio
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[14], mouth[18])
    C = dist.euclidean(mouth[12], mouth[16])
    mar = (A ) / (C)
    return mar

def circularity(eye):
    A = dist.euclidean(eye[1], eye[4])
    radius = A/2.0
    Area = math.pi * (radius ** 2)
    p = 0
    p += dist.euclidean(eye[0], eye[1])
    p += dist.euclidean(eye[1], eye[2])
    p += dist.euclidean(eye[2], eye[3])
    p += dist.euclidean(eye[3], eye[4])
    p += dist.euclidean(eye[4], eye[5])
    p += dist.euclidean(eye[5], eye[0])
    return 4 * math.pi * Area /(p**2)