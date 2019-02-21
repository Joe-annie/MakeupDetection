import os
import cv2
import numpy as np


# 以一张照片为例抽取特征

# test_img_path = '001_1_n.jpgeye_1.jpg'
# main_img = cv2.imread(test_img_path)
#
# # Preprocessing
# img = cv2.cvtColor(main_img, cv2.COLOR_BGR2RGB)
# gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
# blur = cv2.GaussianBlur(gs, (25, 25), 0)
# ret_otsu, im_bw_otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
# kernel = np.ones((50, 50), np.uint8)
# closing = cv2.morphologyEx(im_bw_otsu, cv2.MORPH_CLOSE, kernel)
#
# # Shape features
# image, contours, _ = cv2.findContours(closing, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# cnt = contours[0]
# M = cv2.moments(cnt)
# area = cv2.contourArea(cnt)
# perimeter = cv2.arcLength(cnt, True)
# x, y, w, h = cv2.boundingRect(cnt)
# aspect_ratio = float(w) / h
# rectangularity = w * h / area
# circularity = ((perimeter) ** 2) / area
#
# # Color features
# red_channel = img[:, :, 0]
# green_channel = img[:, :, 1]
# blue_channel = img[:, :, 2]
# blue_channel[blue_channel == 255] = 0
# green_channel[green_channel == 255] = 0
# red_channel[red_channel == 255] = 0
#
# red_mean = np.mean(red_channel)
# green_mean = np.mean(green_channel)
# blue_mean = np.mean(blue_channel)
#
# red_std = np.std(red_channel)
# green_std = np.std(green_channel)
# blue_std = np.std(blue_channel)
#
#
# Texture features
# textures = mt.features.haralick(gs)
# print(textures.shape)
# ht_mean = textures.mean(axis=0)
# contrast = ht_mean[1]
# correlation = ht_mean[2]
# inverse_diff_moments = ht_mean[4]
# entropy = ht_mean[8]



# Solution2：
import lib.Descriptors as des

print("Test start...")
img1 = cv2.imread('001_1_n.jpgeye_2.jpg', 0)


# print ("Testing Color Correlogram")
# img4AutoCorr = cv2.imread(img1, 1)
# matrix = des.autoCorrelogram(img4AutoCorr)
# Klist = [1,3,5,7]
# for idx,k in enumerate(Klist):
#     print ("k = ", k)
#     print (matrix[idx]) #[0.3521739130434783, 0.6478260869565218]

# print("Testing LBP")
# transformed_img = des.lbp(img1)
# print(transformed_img.shape)

# cv2.imshow('image', img1)
# cv2.imshow('thresholded image', transformed_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Texture features

print ("Testing HOG")
hog = des.HoG(img1,See_graph=True)
print ("The output Matrix of HOG: ")
print (hog.ravel().shape)
