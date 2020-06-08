import argparse

def arguments():
    # construct the argument parse and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", default="arguments_file/shape_predictor_68_face_landmarks.dat",
                    help="path to facial landmark predictor")
    ap.add_argument('-m', '--model', default="arguments_file/light_vgg.hdf5",
                    help="path to light_dense_model")
    ap.add_argument("-a", "--alarm", type=str, default="arguments_file/alarm.wav",
                    help="path alarm .WAV file")
    ap.add_argument("-w", "--webcam", type=int, default=0,
                    help="index of webcam on system")

    # Define constants for drowsiness determination.
    # The snow aspect ratio is normalized to the 5 frame mean according to the user.
    # Determine by including mouth aspect ratio, mar / ear ratio, and pupil circularity.
    # The alarm sounds according to the result of the judgment.
    ap.add_argument("-cf", "--CONSEC-FRAMES", type=int, default=60,
                    help="judgement frame")
    ap.add_argument("-cf2", "--CONSEC-FRAMES2", type=int, default=300,
                   help="judgement frame2")
    ap.add_argument("-ear", "--EYE-AR-THRESH", type=float, default=0.045,
                    help="ear thresh")
    ap.add_argument("-cir", "--CIR-THRESH", type=float, default=0.3,
                    help="cir thresh")
    ap.add_argument("-mar", "--MOUTH-AR-THRESH", type=float, default=0.6,
                    help="mar thresh")
    ap.add_argument("-moe", "--MOE-THRESH", type=float, default=1.5,
                    help="moe thresh")

    # construct the argument parser and parse command line arguments
    ap.add_argument("-i", "--ip", type=str, default='192.168.0.42',
                    help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, default='8000',
                    help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-f", "--frame-count", type=int, default=1,
                    help="# of frames used to construct the background model")

    args = vars(ap.parse_args())

    return args