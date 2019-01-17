import glob
import os

from utils import (
    convert_cv2matplot,
    detect_faces,
    mark_faces,
    plot_side_by_side_comparison,
    read_cv_image_from,
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

    image = read_cv_image_from(imagePath)

    faces = detect_faces(image)

    print("Found {0} faces!".format(len(faces)))

    result = mark_faces(image, faces)

    image, result = convert_cv2matplot(image, result)

    plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")
