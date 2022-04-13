# Author:Alvin
# Time:2021/10/27 14:46
import os
import shutil
import time
import dlib
import cv2
detector = dlib.get_frontal_face_detector()
print(type(detector))
