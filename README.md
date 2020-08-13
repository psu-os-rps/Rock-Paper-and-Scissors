# Rock-Paper-and-Scissors
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/psu-os-rps/Rock-Paper-and-Scissors/blob/master/LICENSE)

Copyright (c) 2020 psu-os-rss

## Team members

- Jing Yang
- Boxuan Zhang


## Description

Rock Paper Scissors implements a fingers count pipeline. In this implementation, the images of a hand in different postures and positions are captured by a camera, and the number of fingers are counted and displayed in real time. Functions including `threshold`, `contours` and `convex hull` as well as the distance calculation function from `sklearn` are utilized.


## Project Examples

- [Finger Counters in Real Time](https://github.com/psu-os-rps/Rock-Paper-and-Scissors/blob/master/example/Finger%20Count_result.mp4)
- [Background Threshold](https://github.com/psu-os-rps/Rock-Paper-and-Scissors/blob/master/example/Thresholded_result.mp4)


## Environment

- Hardware: local machine with a camera
- Platform: python 3
- Libraries: OpenCV, sklearn, numpy.


## Pre-requirements

Set up the python library first with following command:
```shell
$ pip install OpenCV-python
$ pip install -U scikit-learn
```

Git Clone and run the python file:
```shell
$ git clone https://github.com/psu-os-rps/Rock-Paper-and-Scissors.git
$ cd Rock-Paper-and-Scissors/src
$ python main.py
```

IMPORTANT: Please run the program with a camera, otherwise it causes an Error.


## Reference:

- [Finger Detection using OpenCV and Python From lzane](https://github.com/lzane/Fingers-Detection-using-OpenCV-and-Python)
- [COVID-19 Face Mask Detector with OpenCV Keras TensorFlow and Deep Learning](https://www.pyimagesearch.com/2020/05/04/covid-19-face-mask-detector-with-opencv-keras-tensorflow-and-deep-learning/)
- [COVID-19 Face Mask Detector with OpenCV Dataset](https://github.com/prajnasb/observations/tree/master/experiements/data)
- [OpenCV Documentation](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html)


## License

Check the [License](https://github.com/psu-os-rps/Rock-Paper-and-Scissors/blob/master/LICENSE) file. This program is under the MIT license.


## Program is running now.

List the future goals and concerns:

### Goals:
- Background Removal (achieved)
- erode and dilate threshold (achieved)
- Next: Change hardcoded parameters to other format with config file or automatic.
- convex hull replaced (Still find better method to counter fingers numbers)
- Use Deep Learning method instead of current Algorithm (Dreaming)
- Build more Detector Features, soundless protector is contributing (Get some ideas, but need more discussion)

### Concerns:
- Background (Light, hardware still becomes concern...)
- erode and dilate threshold (Although fix the threshold problems, the finger counter is not perfect...)
- Waiting for adding new concerns.