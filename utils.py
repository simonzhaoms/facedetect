import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import re
import toolz
import urllib

FACECASCADE = cv.CascadeClassifier("haarcascade_frontalface_default.xml")

MARKCOLOR = (0, 255, 0)  # Green
MARKWIDTH = 4

SCALEFACTOR = 1.2
MINNEIGHBORS = 5
MINSIZE = 30


def _plot_image(ax, img, cmap=None, label=''):
    """Plot <img> in <ax>."""

    ax.imshow(img, cmap)
    ax.tick_params(
        axis='both',
        which='both',
        # bottom='off',  # 'off', 'on' is deprecated in matplotlib > 2.2
        bottom=False,
        # top='off',
        top=False,
        # left='off',
        left=False,
        # right='off',
        right=False,
        # labelleft='off',
        labelleft=False,
        # labelbottom='off')
        labelbottom=False)
    ax.set_xlabel(label)


def plot_side_by_side_comparison(
        leftimg,
        rightimg,
        leftlabel='Original Image',
        rightlabel='Result',
        leftcmap=None,
        rightcmap=None):
    """Plot two images side by side."""

    # Setup canvas

    gs = gridspec.GridSpec(6, 13)
    gs.update(hspace=0.1, wspace=0.001)
    fig = plt.figure(figsize=(7, 3))

    # Plot Left image

    ax = fig.add_subplot(gs[:, 0:6])
    _plot_image(ax, leftimg, cmap=leftcmap, label=leftlabel)

    # Plot right image

    ax = fig.add_subplot(gs[:, 7:13])
    _plot_image(ax, rightimg, cmap=rightcmap, label=rightlabel)

    # Show all of them

    plt.show()


def detect_faces(imagePath, scaleFactor=SCALEFACTOR, minNeighbors=MINNEIGHBORS, minSize=MINSIZE):

    # Read the image

    image = read_cv_image_from(imagePath)

    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Detect faces in the image

    faces = FACECASCADE.detectMultiScale(
        gray,
        scaleFactor=scaleFactor,
        minNeighbors=minNeighbors,
        minSize=(minSize, minSize))

    return image, faces


def mark_faces(image, faces):
    """Mark the <faces> in <image>."""

    result = image.copy()

    # Draw a rectangle around the faces

    for (x, y, w, h) in faces:
        cv.rectangle(result, (x, y), (x + w, y + h), MARKCOLOR, MARKWIDTH)

    return result


def convert_cv2matplot(*images):
    """Convert color space between OpenCV and Matplotlib.

    Because OpenCV and Matplotlib use different color spaces.
    """

    res = []
    for image in images:
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        res.append(image)

    return tuple(res)


def _is_url(url):
    """Check if url is a valid URL."""

    urlregex = re.compile(
        r'^(?:http|ftp)s?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if re.match(urlregex, url) is not None:
        return True
    else:
        return False


def read_cv_image_from(url):
    """Read an image from url or file as grayscale opencv image."""

    return toolz.pipe(
        url,
        urllib.request.urlopen if _is_url(url) else lambda x: open(x, 'rb'),
        lambda x: x.read(),
        bytearray,
        lambda x: np.asarray(x, dtype="uint8"),
        lambda x: cv.imdecode(x, cv.IMREAD_COLOR))
