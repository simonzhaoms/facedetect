import argparse

from utils import (
    convert_cv2matplot,
    detect_faces,
    mark_faces,
    plot_side_by_side_comparison,
    read_cv_image_from,
    FaceParams,
    FACEPARAMS,
)

# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

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
    default=FACEPARAMS.scaleFactor,
    help='scale factor ({} by default, must > 1)'.format(FACEPARAMS.scaleFactor)
)

parser.add_argument(
    '--minNeighbors',
    type=int,
    default=FACEPARAMS.minNeighbors,
    help='minimum neighbors ({} by default, integer, must > 1)'.format(FACEPARAMS.minNeighbors)
)

parser.add_argument(
    '--minSize',
    type=int,
    default=FACEPARAMS.minSize,
    help='minimum size ({} by default, integer, must > 1)'.format(FACEPARAMS.minSize)
)

args = parser.parse_args()

face_params = FaceParams(
    args.scaleFactor,
    args.minNeighbors,
    args.minSize)

# ----------------------------------------------------------------------
# Face detection
# ----------------------------------------------------------------------


image = read_cv_image_from(args.image)

faces = detect_faces(image, face_params=face_params)

print("Found {0} faces!".format(len(faces)))

result = mark_faces(image, faces)

image, result = convert_cv2matplot(image, result)

plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")
