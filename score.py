import argparse

from utils import (
    convert_cv2matplot,
    detect_faces,
    mark_faces,
    plot_side_by_side_comparison,
    read_cv_image_from,
    MIN_NEIGHBORS,
    MIN_SIZE,
    SCALE_FACTOR,
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
    default=SCALE_FACTOR,
    help='scale factor ({} by default, must > 1)'.format(SCALE_FACTOR)
)

parser.add_argument(
    '--minNeighbors',
    type=int,
    default=MIN_NEIGHBORS,
    help='minimum neighbors ({} by default, integer, must > 1)'.format(MIN_NEIGHBORS)
)

parser.add_argument(
    '--minSize',
    type=int,
    default=MIN_SIZE,
    help='minimum size ({} by default, integer, must > 1)'.format(MIN_SIZE)
)

args = parser.parse_args()

# Face detection

image = read_cv_image_from(args.image)

faces = detect_faces(
    image,
    scaleFactor=args.scaleFactor,
    minNeighbors=args.minNeighbors,
    minSize=args.minSize
)

print("Found {0} faces!".format(len(faces)))

result = mark_faces(image, faces)

image, result = convert_cv2matplot(image, result)

plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")

