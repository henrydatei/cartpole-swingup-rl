# https://pyimagesearch.com/2016/02/08/opencv-shape-detection/

from cartpole.Shapedetector import ShapeDetector
import imutils
import cv2
from picamera2 import Picamera2
from libcamera import controls

picam = Picamera2()
picam.preview_configuration.main.format="RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()
picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})

for _ in range(10000):
	image = picam.capture_array()
	# load the image and resize it to a smaller factor so that
	# the shapes can be approximated better
	resized = imutils.resize(image, width=300)
	ratio = image.shape[0] / float(resized.shape[0])
	# convert the resized image to grayscale, blur it slightly,
	# and threshold it
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
	# blurred = cv2.GaussianBlur(gray, (3, 3), 0)
	blurred = gray
	thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]
	# find contours in the thresholded image and initialize the
	# shape detector
	cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
		cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	sd = ShapeDetector()

	# loop over the contours
	for c in cnts:
		# compute the center of the contour, then detect the name of the
		# shape using only the contour
		M = cv2.moments(c)
		if M["m00"] < 1000 and M["m00"] > 0:
			cX = int((M["m10"] / M["m00"]) * ratio)
			cY = int((M["m01"] / M["m00"]) * ratio)
			shape = sd.detect(c)
			cv2.contourArea(c)
			# if (shape == "rectangle" and M["m00"] > 600 and M["m00"] < 850) or (shape == "triangle" and M["m00"] > 400 and M["m00"] < 700):
				# multiply the contour (x, y)-coordinates by the resize ratio,
				# then draw the contours and the name of the shape on the image
			c = c.astype("float")
			c *= ratio
			c = c.astype("int")
			cv2.drawContours(image, [c], -1, (0, 255, 0), 2)
			# cv2.putText(image, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
			cv2.putText(image, str(cv2.contourArea(c)) + ", " + shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

	cv2.imshow("Frame", image)
	key = cv2.waitKey(1) and 0xFF
	if key == ord("q"):
		cv2.destroyAllWindows()
		break

cv2.destroyAllWindows()