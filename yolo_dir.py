import sys
import os
from yolo import YOLO
from PIL import Image
import numpy as np
import cv2

yolo = YOLO()
for root, _, item in os.walk('data/obj/'):
    for file in item:
        if(os.path.splitext(file)[1] != '.png'):
            continue
        #print(file)
        image = Image.open(os.path.join(root,file))
        image = yolo.detect_image(image)
        result = np.asarray(image)
        cv2.namedWindow("result", cv2.WINDOW_NORMAL)
        cv2.imshow("result", result)
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
        if cv2.waitKey(0) & 0xFF == ord('w'):
            continue

yolo.close_session()

