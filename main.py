import numpy as np
import cv2
import time
from picamera2 import *
from null_preview import *
from sty import fg, Style, RgbFg, rs, bg
import os, sys

if sys.platform == "win32":
    os.system('color')

# cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cv2.startWindowThread()

picam2 = Picamera2()
preview = NullPreview(picam2)
picam2.configure(picam2.preview_configuration(main={"format":'XRGB8888', "size": (640,480)}))
picam2.start()

def transform_image(img):
    # print(img.shape)
    if not img.shape[0] or not img.shape[1]:
        return
    img = cv2.resize(img, (50, 50), interpolation = cv2.INTER_AREA)
    
    rgb_matrix = np.zeros((img.shape[0], img.shape[1]))

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            rgb_matrix[x][y] = 0.21 * int(img[y][x][2]) + 0.72 * int(img[y][x][1]) + 0.07 * int(img[y][x][0])

    ascii_matrix = np.full((img.shape[0], img.shape[1]), "")
    ascii_string = "`^\",:;Il!i~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            ascii_matrix[x][y] = ascii_string[int((rgb_matrix[x][y] / 255)*64)] + ascii_string[int((rgb_matrix[x][y] / 255)*64)] + ascii_string[int((rgb_matrix[x][y] / 255)*64)]
            
    for x in range(0, img.shape[0]):
        for y in range(0, img.shape[1]):
            fg.color = Style(RgbFg(int(img[x][y][2]), int(img[x][y][1]), int(img[x][y][0])))
            if (y == img.shape[1] - 1):   
                print(fg.color + ascii_matrix[x][y] + fg.rs)
            else:
                print(fg.color + ascii_matrix[x][y] + fg.rs, end='')

while True:
    # Capture frame-by-frame
    frame = picam2.capture_array()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Detect faces
    
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    if len(faces) == 0:
        print("Waiting to detect a face...")
    else: 
        print(f"Detected {len(faces)}") 
        for (x, y, z, w) in faces:
            #img is numpy array, finds one face
            frame = frame[y:y+w, x:x+z]
            # frame = cv2.resize(frame, (200, 200), interpolation=cv2.INTER_AREA)
            transform_image(frame)

    transform_image(frame)

    # Display the resulting frame
    time.sleep(1)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

