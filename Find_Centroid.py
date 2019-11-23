# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 02:44:57 2019

@author: Shivendra Singh

"""

from __future__ import division

import cv2
import numpy as np
# Input Image
image = cv2.imread("image.jpg")
# Convert to HSV color space
hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
# Define range of white color in HSV
lower_white = np.array([0, 0, 212])
upper_white = np.array([131, 255, 255])
# Threshold the HSV image
mask = cv2.inRange(hsv, lower_white, upper_white)
# Remove noise
kernel_erode = np.ones((4,4), np.uint8)
eroded_mask = cv2.erode(mask, kernel_erode, iterations=1)
kernel_dilate = np.ones((6,6),np.uint8)
dilated_mask = cv2.dilate(eroded_mask, kernel_dilate, iterations=1)
# Find the different contours
im2, contours, hierarchy = cv2.findContours(dilated_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# Sort by area (keep only the biggest one)
contours = sorted(contours, key=cv2.contourArea, reverse=True)[:1]
if len(contours) > 0:
    M = cv2.moments(contours[0])
    # Centroid
    cx = int(M['m10']/M['m00'])
    cy = int(M['m01']/M['m00'])
    print("Centroid of the biggest area: ({}, {})".format(cx, cy))
else:
    print("No Centroid Found")