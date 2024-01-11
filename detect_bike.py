# import the necessary packages
from imutils.video import VideoStream
import numpy as np
import sys
from PIL import Image
import argparse
import imutils
import time
import cv2
import os,glob

num = 1
def background_sub(c_dup,crop):
	def trim(frame):
		global count
		#crop top
		if not np.sum(frame[0]):
			count+=1
			return trim(frame[1:])
		return frame

	global count
	count=0
	gray = cv2.cvtColor(c_dup, cv2.COLOR_BGR2GRAY) 
	ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV +cv2.THRESH_OTSU)
	image = trim(thresh)
	if(count>10):
		img_resized = crop[count-10:]
	elif(count>5 and count<=10):
		img_resized = crop[count-4:]
	else:
		img_resized = crop[:]
	return img_resized


# initialize the list of class labels MobileNet SSD was trained to
# detect, then generate a set of bounding box colors for each class
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

IGNORE = set(["motorbike"])

COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

model = "MobileNetSSD_deploy.caffemodel"
prototxt = "MobileNetSSD_deploy.prototxt.txt"
conf = 0.2
# load our serialized model from disk
print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe(prototxt, model)

# initialize the video stream, allow the cammera sensor to warmup,
print("[INFO] starting video stream...")

vs = cv2.VideoCapture('videos/1.mp4')

#time.sleep(2.0)

detected_objects = []
kill=0
# loop over the frames from the video stream
loop = 0
while True:
    ret, fr = vs.read()
    if np.any(fr == None):
        continue
    else:
        frame = np.array(fr, dtype=np.uint8)
        
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = imutils.resize(frame, width=800)
    
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    if loop % 12 == 0:
        for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > conf:
                idx = int(detections[0, 0, i, 1])
                
                if CLASSES[idx] not in IGNORE or confidence < 0.30:
                    continue
                else:
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    cropped = frame[startY-75:endY-5, startX+30:endX-30]
                    cropped_dup = frame[startY-75:endY-5, startX+60:endX-60]

                    p1 = './images/bike_'+str(num)+'.png'
                    p2 = './images/bike_'+str(num)+'_'+str(num)+'.png'

                    if len(cropped_dup) != 0:
                        try:
                            img1 = Image.fromarray(cropped_dup, 'RGB')
                            img2 = Image.fromarray(cropped, 'RGB')
                            img1.save(p1)
                            img2.save(p2)
                            im_p1 = cv2.imread(p1)
                            im_p2 = cv2.imread(p2)
                            num += 1
                        except:
                            print("Error")
                        img_resized = background_sub(im_p1, im_p2)
                        cv2.imwrite(p1, img_resized)

                    label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                    cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[idx], 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

    loop += 1
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# Release video stream and close OpenCV windows
vs.release()
cv2.destroyAllWindows()