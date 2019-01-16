import argparse

from utils import (
    plot_side_by_side_comparison,
    detect_faces,
    mark_faces,
    convert_cv2matplot,
)

# Parse command line arguments

parser = argparse.ArgumentParser(
    prog='score',
    description='Detect faces in an image.'
)

parser.add_argument(
    'image',
    type=str,
    help='image path or URL'
)

parser.add_argument(
    '--scaleFactor',
    type=float,
    default=1.2,
    help='scale factor (1.2 by default, must > 1)'
)

parser.add_argument(
    '--minNeighbors',
    type=int,
    default=5,
    help='minimum neighbors (5 by default, integer, must > 1)'
)

parser.add_argument(
    '--minSize',
    type=int,
    default=30,
    help='minimum size (30 by default, integer, must > 1)'
)

args = parser.parse_args()

# Face detection

image, faces = detect_faces(
    args.image,
    scaleFactor=args.scaleFactor,
    minNeighbors=args.minNeighbors,
    minSize=args.minSize
)

print("Found {0} faces!".format(len(faces)))

result = mark_faces(image, faces)

image, result = convert_cv2matplot(image, result)

plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")

