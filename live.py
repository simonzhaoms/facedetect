import argparse
import cv2 as cv
import matplotlib.pyplot as plt
import sys

from matplotlib.animation import FuncAnimation
from utils import (
    get_faces_frame,
    FaceParams,
    option_parser,
)

# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog='live',
    parents=[option_parser],
    description='Detect faces from camera.'
)

args = parser.parse_args()

face_params = FaceParams(
    args.scaleFactor,
    args.minNeighbors,
    args.minSize)

# ----------------------------------------------------------------------
# Setup video
# ----------------------------------------------------------------------

camera = cv.VideoCapture(0)  # Get the first camera
if not camera.isOpened():
    print('Unable to load camera!')
    sys.exit(1)

# ----------------------------------------------------------------------
# Setup plot
# See [update frame in matplotlib with live camera preview](https://stackoverflow.com/a/44604435)
# ----------------------------------------------------------------------

plt.axis('off')  # Turn off axis in plot window
plt.gcf().canvas.mpl_connect(  # Set plot window close event
    "key_press_event",
    lambda event: plt.close(event.canvas.figure) if event.key == 'q' else None
)

# ----------------------------------------------------------------------
# Detect faces and show results
# See [update frame in matplotlib with live camera preview](https://stackoverflow.com/a/44604435)
# ----------------------------------------------------------------------

print("\nPlease type 'q' to quit.")

im = plt.gca().imshow(get_faces_frame(camera, face_params=face_params))
video = FuncAnimation(
    plt.gcf(),
    lambda i: im.set_data(get_faces_frame(camera, face_params=face_params)),  # Update plot window with new camera frame
    interval=100)

plt.show()

# When everything is done, release the capture

camera.release()
