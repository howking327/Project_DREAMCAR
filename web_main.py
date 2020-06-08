import threading
import time
from flask import Flask, Response, render_template ,request
from argument_parser import arguments
from detecting import driver_detect, f_generate, p_generate

# argparse arguments function call
args = arguments()

# initialize a flask object
app = Flask(__name__)

global t

@app.route("/")
def index():
	# return the rendered templat
	return render_template("index.html")

@app.route("/modelon" , methods=['POST'])
def model_on():
    mode = request.form['mode']
    return_msg = "fail"

    if mode == "on":
        print('Thread On')
        # start a thread that will perform motion detection
        t = threading.Thread(target=driver_detect)
        t.daemon = True
        t.start()
    else:
        pass

    while True :
        time.sleep(1)
        if not t.is_alive():
            return_msg = "success"
            break
    return return_msg

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route("/video_feed")
def video_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(f_generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=frame")

@app.route("/prob_feed")
def prob_feed():
    # return the response generated along with the specific media
    # type (mime type)
    return Response(p_generate(),
                    mimetype = "multipart/x-mixed-replace; boundary=canvas")

# check to see if this is the main thread of execution
if __name__ == '__main__':
    # start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)