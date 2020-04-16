import cv2
import glob
import numpy as np


# function to compute new warping parameters
def affineLKtracker(image, tmp, rect):

    # get perspective matrix
    M = cv2.getPerspectiveTransform(rect, dst)

    # get warped image
    W = cv2.warpPerspective(image, M, (122, 89))

    # difference of template and warped image
    diff = cv2.subtract(tmp, W)

    # display warped image
    cv2.imshow('warped image', W)
    # print(W.shape)
    cv2.imshow('difference', diff)


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
    box = np.array([[67, 51], [189, 52], [189, 139], [67, 138]], dtype='float32')

    # destination point coordinates for warping
    dst = np.array([[0, 0], [122, 0], [122, 89], [0, 89]], dtype='float32')

    for img in images:

        affineLKtracker(img, template, box)
        # cv2.imshow('image', img)
        cv2.waitKey(0)
