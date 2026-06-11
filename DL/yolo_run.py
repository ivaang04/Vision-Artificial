#!/usr/bin/env python

from ultralytics import YOLO

model = YOLO("mando-ultima-prueba.pt")

import cv2 as cv
from umucv.stream import autoStream

for key,frame in autoStream():


    [result] = model(frame)

    cv.imshow("YOLO", result.plot())

