
import cv2
import os
import numpy as np

for i in range(100):
    #TODO CleanUP the code
    #
    i = i + 1
    directory = './img/test'
    filename = 'Masked_%s.png' %i
    filename = os.path.join('%s' % directory, '%s' % filename)

    img = cv2.imread('./img/test/Generated_Image_%s.png' %i)
    img1 = cv2.imread('./img/test/Input_Image_%s.png' %i)
    _, mask = cv2.threshold(img, thresh=180, maxval=255, type=cv2.THRESH_BINARY)
    im_thresh_gray = cv2.bitwise_and(img1, mask)

    cv2.imwrite(filename, im_thresh_gray)
