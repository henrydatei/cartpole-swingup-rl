#from mimetypes import init
#from picamera.array import PiRGBArray # Generates a 3D RGB array
#from picamera import PiCamera # Provides a Python interface for the RPi Camera Module
import time # Provides time-related functions
import cv2 # OpenCV library
import numpy as np
import math
import time
import datetime
from picamera2 import Picamera2, Preview
from libcamera import controls
picam = Picamera2()

#picam.preview_configuration.main.size=(640,480)
picam.preview_configuration.main.format="RGB888"
picam.preview_configuration.align()
picam.configure("preview")
picam.start()
#picam.set_controls({"AfMode":controls.AfModeEnum.Continuous})

yellow=np.uint8([[[2,60,40]]])  #find using mouse on sticker when cv.imshow
hsv_y=cv2.cvtColor(yellow,cv2.COLOR_BGR2HSV)
print(hsv_y)

#exit()

# Initialize the camera
#camera = PiCamera()
 
# Set the camera resolution
#camera.resolution = (640, 480)
 
# Set the number of frames per second
#camera.framerate = 32
 
# Generates a 3D RGB array and stores it in rawCapture
#raw_capture = PiRGBArray(camera, size=(640, 480))
 
# Wait a certain number of seconds to allow the camera time to warmup
time.sleep(0.1)
 
# Capture frames continuously from the camera
radii=[]
move=0
time1=[]
anglelist=[]
angle=0
count=0   
#camera=cv2.VideoCapture(0)
initime=time.time()
#camera.set(cv2.CAP_PROP_FRAME_HEIGHT,640)
#camera.set(cv2.CAP_PROP_FRAME_WIDTH,480)
#camera.set(cv2.CAP_PROP_FPS,32)
#writer=cv2.VideoWriter('test8.avi',cv2.VideoWriter_fourcc('M','J','P','G'),32,(640,480))
#for frame in camera.capture_continuous(raw_capture, format="bgr", use_video_port=True):
for z in range(10000) :
    count+=1
    time1.append(time.time())
    # Grab the raw NumPy array representing the image
    initime=time.time()
    
    
    #_,image = camera.read()
    image=picam.capture_array()
    
    #rows,cols,_ = image.shape
    #writer.write(image)
    #print('time1:',time.time()-initime)
    
    x_medium=320
    y_medium=240
    
    x1=200
    y1=200
    radius=10
    center=150
    centre=240
    #initime=time.time()
    
    hsv_frame=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    #red color
    #low_red = np.array([161,155,84])
    #high_red = np.array([179,255,255])
    #red_mask=cv2.inRange(hsv_frame,low_red,high_red)
    
    #blue color
    #low_blue = np.array([110,50,50])
    #high_blue = np.array([130,255,255])
    #blue_mask=cv2.inRange(hsv_frame,low_blue,high_blue)
    
    #blue color
    low_yellow = np.array([20,180,70])
    high_yellow = np.array([35,255,240])
    yellow_mask=cv2.inRange(hsv_frame,low_yellow,high_yellow)
       
    #purple mask
    low_pur = np.array([150,95,85])
    high_pur = np.array([180,255,255])
    pur_mask=cv2.inRange(hsv_frame,low_pur,high_pur)
       
    contours_y, _ = cv2.findContours(yellow_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours_y, key=lambda x:cv2.contourArea(x), reverse=True)
    
    if len(contours_y)==0:
        contours_p, _ = cv2.findContours(pur_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        contours = sorted(contours_p, key=lambda x:cv2.contourArea(x), reverse=True)
        
    for cnt in contours:
        
        (x, y, w, h) = cv2.boundingRect(cnt)
        #in case of circle contours
        #((x1,y1),radius)=cv2.minEnclosingCircle(cnt)
        #radii.append(radius)
        #initime=time.time()
        #M = cv2.moments(cnt)
        #center=(int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"]))

        #cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
        x_medium = int((x + x + w) / 2)
        y_medium = int((y + y + h) / 2)
        break
   
    x_ori=296
    y_ori=405
    print(x_medium,y_medium)
    #cv2.line(image ,(x_medium, 0), (x_medium, 480), (0,0,255), 2)
    #cv2.line(image ,(0, y_medium), (640, y_medium), (0,0,255), 2)
    #if radius >20 and radius <30:
    #cv2.line(image ,(x_ori, y_ori), (x_medium,y_medium), (255,0,255), 2)
   # cv2.circle(image, (int(x1),int(y1)), int(radius), (0, 0, 255), 2)
    #cv2.circle(image, center, 5, (0, 0, 255), -1)
    

    if y_medium==y_ori and x_medium>x_ori:
        print("X+")
        
        angle=math.pi/2
    elif y_medium==y_ori and x_medium<x_ori:
        print("X-")
        
        angle=1.5*math.pi
    elif y_medium>y_ori and x_medium==x_ori:
        print("Y-")
        
        angle=math.pi
    elif y_medium<y_ori and x_medium==x_ori:
        print("Y+")
        
        angle=0
    
    elif y_medium<y_ori and x_medium>x_ori:
        print("Q1")
        angle=math.atan((x_medium-x_ori)/(y_ori-y_medium))
    
    elif y_medium>y_ori and x_medium>x_ori:
        print("Q2")
        
        angle=math.pi-math.atan((x_medium-x_ori)/(y_medium-y_ori))
    
    elif y_medium>y_ori and x_medium<x_ori:
        print("Q3")
        
        angle=math.pi+math.atan((x_ori-x_medium)/(y_medium-y_ori))
    
    else:
        print("Q4")
        
        angle=2*math.pi-math.atan((x_ori-x_medium)/(y_ori-y_medium))
    
    if len(contours_y)==0:
        angle=math.pi+angle
        if angle > 2*math.pi:
            angle=angle-2*math.pi
            
    #print('angle:',math.degrees(angle))
    
    
    anglelist.append(angle)
    
    #print(x_medium,y_medium)
    # Display the frame using OpenCV
    cv2.imshow("Frame", image)
    #cv2.imshow('red mask',red_mask)
    #cv2.imshow('yellow mask',yellow_mask)
    
    # Wait for keyPress for 1 millisecond
    key = cv2.waitKey(1) and 0xFF
    t2=time.time()
    print('time:',t2-initime)
    
     
    # Clear the stream in preparation for the next frame
    #raw_capture.truncate(0)
    
    #if count>1:
     #   thetadot=(anglelist[count-1]-anglelist[count-2])/(time1[count-1]-time1[count-2])
      #  print(angle,thetadot)
        #break
    
    
   # if angle>15 or angle <-15:
        
        #cv2.destroyAllWindows()    
        
       # break
    # If the `q` key was pressed, break from the loop
    if key == ord("q"):
        cv2.destroyAllWindows()
        break

#camera.release()
#writer.release()
cv2.destroyAllWindows()
#frame.release()

    

