import numpy as np
import cv2
img = np.zeros((28, 28), dtype=np.uint8)
cv2.putText(img, '5', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 255, 2)
cv2.imwrite('test_5.png', img)
