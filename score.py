import argparse

from utils import (
    plot_side_by_side_comparison,
    detect_faces,
    mark_faces,
    convert_cv2matplot,
    SCALEFACTOR,
    MINNEIGHBORS,
    MINSIZE,
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
    default=SCALEFACTOR,
    help='scale factor ({} by default, must > 1)'.format(SCALEFACTOR)
)

parser.add_argument(
    '--minNeighbors',
    type=int,
    default=MINNEIGHBORS,
    help='minimum neighbors ({} by default, integer, must > 1)'.format(MINNEIGHBORS)
)

parser.add_argument(
    '--minSize',
    type=int,
    default=MINSIZE,
    help='minimum size ({} by default, integer, must > 1)'.format(MINSIZE)
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

