import argparse

from utils import (
    convert_cv2matplot,
    detect_faces,
    FaceParams,
    get_abspath,
    is_url,
    mark_faces,
    option_parser,
    plot_side_by_side_comparison,
    read_cv_image_from,
)

# ----------------------------------------------------------------------
# Parse command line arguments
# ----------------------------------------------------------------------

parser = argparse.ArgumentParser(
    prog='score',
    parents=[option_parser],
    description='Detect faces in an image.'
)

parser.add_argument(
    'image',
    type=str,
    help='image path or URL'
)

args = parser.parse_args()

# Wrap face detection parameters.

face_params = FaceParams(
    args.scaleFactor,
    args.minNeighbors,
    args.minSize)

# ----------------------------------------------------------------------
# Face detection
# ----------------------------------------------------------------------


image = read_cv_image_from(args.image if is_url(args.image) else get_abspath(args.image))

faces = detect_faces(image, face_params=face_params)

print("Found {0} faces!".format(len(faces)))

result = mark_faces(image, faces)

image, result = convert_cv2matplot(image, result)

plot_side_by_side_comparison(image, result, rightlabel="Detected Faces")
