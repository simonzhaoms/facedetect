import glob
import os

from utils import (
    plot_side_by_side_comparison,
    detect_faces,
    mark_faces,
    convert_cv2matplot,
)

# Setup

IMG_PATH = 'images'

cwd = os.getcwd()
print("Demonstrate face detection using images found in\n{}\n".format(
    os.path.join(cwd, IMG_PATH)
))

print("Please close each image (Ctrl-w) to proceed through the demonstration.\n")

imagePaths = glob.glob(os.path.join(IMG_PATH, "*"))
imagePaths.sort()

for imagePath in imagePaths:

    image, faces = detect_faces(imagePath)

    print("Found {0} faces!".format(len(faces)))

    result = mark_faces(image, faces)

    image, result = convert_cv2matplot(image, result)

    plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")
