# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 22:20:32 2020

@author: gyans
"""

# import the necessary packages
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import os
from pyimagesearch.motion_detection import SingleMotionDetector
from imutils.video import VideoStream
from flask import Response
from flask import Flask
from flask import render_template
import threading
import argparse
import datetime
import imutils
import time
import cv2

# initialize the output frame and a lock used to ensure thread-safe
# exchanges of the output frames (useful for multiple browsers/tabs
# are viewing tthe stream)
outputFrame = None
lock = threading.Lock()

# initialize a flask object
app = Flask(__name__)

# initialize the video stream and allow the camera sensor to
# warmup
#vs = VideoStream(usePiCamera=1).start()
vs = VideoStream(src=0).start()
time.sleep(2.0)

@app.route("/")
def index():
	# return the rendered template
	return render_template("index1.html")
def detect_and_predict_mask(frame, faceNet, maskNet):
    	# grab the dimensions of the frame and then construct a blob
    	# from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
    		(104.0, 177.0, 123.0))
    
    	# pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    
    	# initialize our list of faces, their corresponding locations,
    	# and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []
    
    	# loop over the detections
    for i in range(0, detections.shape[2]):
    		# extract the confidence (i.e., probability) associated with
    		# the detection
    	confidence = detections[0, 0, i, 2]
    
    		# filter out weak detections by ensuring the confidence is
    		# greater than the minimum confidence
    	if confidence > args["confidence"]:
    			# compute the (x, y)-coordinates of the bounding box for
    			# the object
    		box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
    		(startX, startY, endX, endY) = box.astype("int")
    
    			# ensure the bounding boxes fall within the dimensions of
    			# the frame
    		(startX, startY) = (max(0, startX), max(0, startY))
    		(endX, endY) = (min(w - 1, endX), min(h - 1, endY))
    
    			# extract the face ROI, convert it from BGR to RGB channel
    			# ordering, resize it to 224x224, and preprocess it
    		face = frame[startY:endY, startX:endX]
    		face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    		face = cv2.resize(face, (224, 224))
    		face = img_to_array(face)
    		face = preprocess_input(face)
    		face = np.expand_dims(face, axis=0)
    
    			# add the face and bounding boxes to their respective
    			# lists
    		faces.append(face)
    		locs.append((startX, startY, endX, endY))
    
    	# only make a predictions if at least one face was detected
    if len(faces) > 0:
    		# for faster inference we'll make batch predictions on *all*
    		# faces at the same time rather than one-by-one predictions
    		# in the above `for` loop
    	preds = maskNet.predict(faces)
    
    	# return a 2-tuple of the face locations and their corresponding
    	# locations
    return (locs, preds)

def detect_motion(frameCount,face,model):
	# grab global references to the video stream, output frame, and
	# lock variables
    global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face,
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    maskNet = load_model(model)
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")

	# loop over frames from the video stream
    while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		

		# grab the current timestamp and draw it on the frame
        for (box, pred) in zip(locs, preds):
    		# unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
    
    		# determine the class label and color we'll use to draw
    		# the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
    		# display the label and bounding box rectangle on the output
    		# frame
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
        if total > frameCount:
			# detect motion in the image
            motion = md.detect(frame)

			# cehck to see if motion was found in the frame
            if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
        md.update(frame)
        total += 1

		# acquire the lock, set the output frame, and release the
		# lock
        with lock:
            outputFrame = frame.copy()
            
def detect_motion(frameCount,face,model):
	# grab global references to the video stream, output frame, and
	# lock variables
    global vs, outputFrame, lock

	# initialize the motion detector and the total number of frames
	# read thus far
    md = SingleMotionDetector(accumWeight=0.1)
    total = 0
    prototxtPath = os.path.sep.join([face, "deploy.prototxt"])
    weightsPath = os.path.sep.join([face,
    	"res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)
    
    # load the face mask detector model from disk
    #print("[INFO] loading face mask detector model...")
    maskNet = load_model(model)
    
    # initialize the video stream and allow the camera sensor to warm up
    print("[INFO] starting video stream...")

	# loop over frames from the video stream
    while True:
		# read the next frame from the video stream, resize it,
		# convert the frame to grayscale, and blur it
        frame = vs.read()
        frame = imutils.resize(frame, width=600)
        (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)
		

		# grab the current timestamp and draw it on the frame
        for (box, pred) in zip(locs, preds):
    		# unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (mask, withoutMask) = pred
    
    		# determine the class label and color we'll use to draw
    		# the bounding box and text
            label = "Mask" if mask > withoutMask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
    
    		# include the probability in the label
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    
    		# display the label and bounding box rectangle on the output
    		# frame
            cv2.putText(frame, label, (startX, startY - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
		# if the total number of frames has reached a sufficient
		# number to construct a reasonable background model, then
		# continue to process the frame
        if total > frameCount:
			# detect motion in the image
            motion = md.detect(frame)

			# cehck to see if motion was found in the frame
            if motion is not None:
				# unpack the tuple and draw the box surrounding the
				# "motion area" on the output frame
                (thresh, (minX, minY, maxX, maxY)) = motion
                cv2.rectangle(frame, (minX, minY), (maxX, maxY),(0, 0, 255), 2)
		
		# update the background model and increment the total number
		# of frames read thus far
        md.update(frame)
        total += 1

		# acquire the lock, set the output frame, and release the
		# lock
        with lock:
            outputFrame = frame.copy()
		
def generate():
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
		yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
			bytearray(encodedImage) + b'\r\n')

@app.route("/video_feedd")
def video_feedd():
	# return the response generated along with the specific media
	# type (mime type)
	return Response(generate(),
		mimetype = "multipart/x-mixed-replace; boundary=frame")

# check to see if this is the main thread of execution
if __name__ == '__main__':
	# construct the argument parser and parse command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-f", "--face", type=str,
    	default="face_detector",
    	help="path to face detector model directory")
    ap.add_argument("-m", "--model", type=str,
    	default="mask_detector1.model",
    	help="path to trained face mask detector model")
    ap.add_argument("-c", "--confidence", type=float, default=0.5,
    	help="minimum probability to filter weak detections")
    ap.add_argument("-i", "--ip", type=str, 
		help="ip address of the device")
    ap.add_argument("-o", "--port", type=int, 
		help="ephemeral port number of the server (1024 to 65535)")
    ap.add_argument("-e", "--frame-count", type=int, default=100000,
		help="# of frames used to construct the background model")
    args = vars(ap.parse_args())
    
    t = threading.Thread(target=detect_motion, args=(
		args["frame_count"],args["face"],args["model"]))
    t.daemon = True
    t.start()

	# start the flask app
    app.run(host=args["ip"], port=args["port"], debug=True,
            threaded=True, use_reloader=False)

# release the video stream pointer
vs.stop()