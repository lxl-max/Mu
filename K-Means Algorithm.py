"""
1. run
2. choose ROIs, enter "space" to confirm, enter "Esc" to exit
3. get results

library:
pip install opencv-python

parameters:
rois: selected ROIs(RGB)
rgb_roi: one of rois(RGB)
seg_roi: segmented rois
"""

import time
import matplotlib.pyplot as plt
import numpy as np
import cv2

def ms(s):
    return f"{s*1000:.1f} ms"

# Read image
sample_image = cv2.imread('input the path of the figure')
img = cv2.cvtColor(sample_image,cv2.COLOR_BGR2RGB)

# Select ROIs
rects = cv2.selectROIs("select", img, showCrosshair=True, fromCenter=False)
cv2.destroyWindow("select")
rois = []
for i, (x, y, w, h) in enumerate(rects):
    if w > 0 and h > 0:
        roi = img[y:y+h, x:x+w]
        rois.append(roi)

# Show initial image
plt.imshow(img)
plt.title("Initial Image")
plt.axis("off")
plt.show()

# Show segment image
twoDimage = img.reshape((-1, 3))
twoDimage = np.float32(twoDimage)

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
K = 3
attempts = 10

t0 = time.perf_counter()
ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
center = np.uint8(center)
res = center[label.flatten()]
seg_img = res.reshape((img.shape))
t_seg_full = time.perf_counter() - t0

fig, ax = plt.subplots()
plt.imshow(seg_img)
plt.title("Image with K-Means Algorithm")
fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
         ha='center', va='bottom')
plt.axis('off')
plt.show()

# Show ROIs
for i in range(len(rois)):
    plt.imshow(rois[i])
    plt.title(f"ROI {i+1}")
    plt.axis('off')
    plt.show()

# Show segment ROIs
for i in range(len(rois)):

    rgb_roi = rois[i]
    twoDimage = rois[i].reshape((-1, 3))
    twoDimage = np.float32(twoDimage)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 3
    attempts = 10

    t0 = time.perf_counter()
    ret, label, center = cv2.kmeans(twoDimage, K, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    seg_roi = res.reshape((rgb_roi.shape))
    t_seg_full = time.perf_counter() - t0

    fig, ax = plt.subplots()
    plt.imshow(seg_roi)
    plt.title(f"ROI {i+1} with K-Means Algorithm")
    fig.text(0.5, 0.02, f"Execution times: {ms(t_seg_full)}",
             ha='center', va='bottom')
    plt.axis('off')
    plt.show()
