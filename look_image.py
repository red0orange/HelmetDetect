import os
import numpy as np
import cv2


if __name__ == '__main__':
    image_path = "eyes_template.png"

    image = cv2.imread(image_path)
    cv2.imshow("image", image)
    cv2.waitKey(0)
    pass