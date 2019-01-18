# Simple Face Detection #

This is a simple face detection example of using machine learning
algorithms to search faces within a picture.  It originates from
Shantnu Tiwari's tutorial -- [Face Recognition with Python, in Under
25 Lines of
Code](https://realpython.com/face-recognition-with-python/) and [Face
Detection in Python Using a
Webcam](https://realpython.com/face-detection-in-python-using-a-webcam/).
It uses [OpenCV](https://opencv.org) cascade to break the problem of
detecting faces into multiple stages.  The algorithm starts at the top
left of a picture and moves down across small blocks of data.  During
the moves, a series of coarse-to-fine quick tests are carried out on
each block.  And it will only detect a face if all stages pass.

See the github repository for examples of its usage:
https://github.com/simonzhaoms/facedetect


## Usage ##

* To install and demostrate the algorithm:

  ```console
  $ pip3 install mlhub
  $ ml install   facedetect
  $ ml configure facedetect
  $ ml demo      facedetect
  ```

## Examples

To detect faces:

  - From a local image file:

    ```console
    $ ml score facedetect ~/.mlhub/facedetect/images/abba.png
    ```

  - From an image on the web:

    ```console
    $ ml score facedetect https://github.com/opencv/opencv/raw/master/samples/data/lena.jpg
    ```
	
  - From your camera:
  
    ```console
	$ ml live facedetect
	```

Sometimes the algorithm will fail to detect real faces, then you need
to fine-tune the parameters to get the ideal results:

```console
$ ml score facedetect https://github.com/ageitgey/face_recognition/raw/master/tests/test_images/obama.jpg
$ ml score facedetect https://github.com/ageitgey/face_recognition/raw/master/tests/test_images/obama.jpg --scaleFactor 1.3
$ ml live facedetect --scaleFactor 1.3 --minSize 7 --minNeighbors 40
```
