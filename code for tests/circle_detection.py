# https://how2electronics.com/shape-based-object-tracking-with-opencv-on-raspberry-pi/

import cv2
import numpy as np
import time
from picamera2 import Picamera2
from libcamera import controls
 
CAMERA_DEVICE_ID = 0
IMAGE_WIDTH = 320
IMAGE_HEIGHT = 240
fps = 0
 
def set_camera_properties(cap, width, height):
    """ Set resolution properties for the camera """
    cap.set(3, width)
    cap.set(4, height)
 
def capture_frame(picam):
    """ Capture a frame from the video source """
    frame = picam.capture_array()
    # if not frame:
    #     raise ValueError("Failed to read a frame from the camera")
    return frame
 
def detect_circles(gray_frame):
    """ Detect circles using Hough transform and return the circles found """
    return cv2.HoughCircles(gray_frame, cv2.HOUGH_GRADIENT, 1.2, 100)
 
def process_frame(frame):
    """ Blur, convert to grayscale and detect circles """
    # frame = cv2.blur(frame, (3, 3))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    circles = detect_circles(gray)
    return frame, gray, circles
 
def draw_circles_on_frame(frame, circles):
    """ Draw circles and center rectangles on the given frame """
    output = frame.copy()
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(output, (x, y), r, (0, 255, 0), 4)
            cv2.rectangle(output, (x - 5, y - 5), (x + 5, y + 5), (0, 128, 255), -1)
    return output
 
def visualize_fps(image, fps: float) -> np.ndarray:
    """Overlay the FPS value onto the given image."""
    if len(np.shape(image)) < 3:
        text_color = (255, 255, 255)  # white
    else:
        text_color = (0, 255, 0)  # green
    
    row_size = 20  # pixels
    left_margin = 24  # pixels
    font_size = 1
    font_thickness = 1
    
    fps_text = 'FPS: {:.1f}'.format(fps)
    text_location = (left_margin, row_size)
    cv2.putText(image, fps_text, text_location, cv2.FONT_HERSHEY_PLAIN,
                font_size, text_color, font_thickness)
 
    return image
 
def main():
    try:
        picam = Picamera2()
        picam.preview_configuration.main.format="RGB888"
        picam.preview_configuration.align()
        picam.configure("preview")
        picam.start()
        picam.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        # if not cap.isOpened():
        #     raise ValueError("Could not open the camera")
        # set_camera_properties(cap, IMAGE_WIDTH, IMAGE_HEIGHT)
 
        print("Press 'Esc' to exit...")
        
        fps = 0  # Initialize the fps variable
 
        while True:
            start_time = time.time()
            
            frame = capture_frame(picam)
            frame, _, circles = process_frame(frame)
            print(circles)
            output = draw_circles_on_frame(frame, circles)
 
            end_time = time.time()
            seconds = end_time - start_time
            fps = 1.0 / seconds
 
            # Overlay FPS and display frames
            cv2.imshow("Frame", np.hstack([visualize_fps(frame, fps), visualize_fps(output, fps)]))
 
            if cv2.waitKey(33) == 27:  # Escape key
                break
 
    except Exception as e:
        print(e)
 
    finally:
        cv2.destroyAllWindows()
        # cap.release()
 
if __name__ == "__main__":
    main()