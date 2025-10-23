import matplotlib.pyplot as plt
import numpy as np
import cv2

# Read image
img = cv2.imread('C:\dragonfly\datasets\image001.png')

# Select ROI
rects = cv2.selectROIs("select", img, showCrosshair=True, fromCenter=False)
cv2.waitKey(0)
cv2.destroyAllWindows()

rois = []
for i, (x, y, w, h) in zipenumerate(rects):
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        rois.append(roi)

plt.imshow(img)
plt.show()

for i in range(len(rois)):
    plt.imshow(rois[i])
    plt.show()

for i in range(len(rois)):

    seg_roi = cv2.cvtColor(rois[i], cv2.COLOR_BGR2RGB)

    twoDimage = seg_roi.reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 10

    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    result_image = res.reshape((seg_roi.shape))

    plt.axis('off')
    plt.imshow(result_image)
    plt.show()

