import cv2
import glob
import numpy as np


# function to compute new warping parameters
def affineLKtracker(image, tmp, rect, pprev):

    # gradient x and y
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    # get perspective matrix
    warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

    # get warped image
    warp_image = cv2.warpAffine(image, warp_mat, (tmp.shape[1], tmp.shape[0]))
    warp_image = warp_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

    # get 2D rotation matrix
    # ptemp =

    # difference of template and warped image
    # diff = cv2.subtract(tmp, warp_image)

    # display warped image
    cv2.imshow('warped image', warp_image)

    # display gradient image
    cv2.imshow('gradient_X', sobelx)
    cv2.imshow('gradient_Y', sobely)


# main function
if __name__ == '__main__':

    # get images from folder
    filenames = glob.glob("Dataset/Car4/img/*.jpg")
    filenames.sort()
    images = [cv2.imread(img, 0) for img in filenames]

    # get template from folder
    template = cv2.imread('Dataset/templates/croppedCar.png', 0)
    # print(template.shape)

    # bounding box coordinates in image
    box = np.array([[67, 51], [189, 139]], dtype='int32')
    # box = np.array([[128, 52], [189, 139], [67, 138]], dtype='float32')

    # destination point coordinates for warping
    # dst = np.array([[0, 0], [template.shape[1], 0], [template.shape[1], template.shape[0]], [0, template.shape[0]]], dtype='float32')
    # dst = np.array([[0, template.shape[1]*0.33], [template.shape[1]*0.85, template.shape[0]*0.25], [template.shape[1]*0.15, template.shape[0]*0.7]], dtype='float32')

    # warping parameters
    param = np.zeros(6)

    for img in images:

        affineLKtracker(img, template, box, param)
        # cv2.imshow('image', img)
        cv2.waitKey(0)
