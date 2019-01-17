import cv2 as cv
import matplotlib.pyplot as plt
import sys

from matplotlib.animation import FuncAnimation
from utils import get_faces_frame

# Setup video

camera = cv.VideoCapture(0)  # Get the first camera
if not camera.isOpened():
    print('Unable to load camera!')
    sys.exit(1)

# Setup plot

plt.axis('off')  # Turn off axis in plot window
plt.gcf().canvas.mpl_connect(  # Set plot window close event
    "key_press_event",
    lambda event: plt.close(event.canvas.figure) if event.key == 'q' else None
)

# Detect faces and show results

im = plt.gca().imshow(get_faces_frame(camera))
video = FuncAnimation(
    plt.gcf(),
    lambda i: im.set_data(get_faces_frame(camera)),  # Update plot window with new camera frame
    interval=100)

plt.show()

# When everything is done, release the capture

camera.release()

