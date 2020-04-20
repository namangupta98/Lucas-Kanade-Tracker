import cv2
import glob
import numpy as np


# function to compute new warping parameters
def affineLKtracker(image, tmp, rect, pprev):

    # get warping parameters
    warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

    norm_p = 10

    while norm_p > 0.3:

        # get warped image
        warp_image = cv2.warpAffine(image, warp_mat, (0, 0))
        warp_image = warp_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # difference of template and warped image
        diff = cv2.subtract(tmp, warp_image).reshape(-1, 1)

        # gradient x and y
        sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
        sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

        # get sobel warped image
        warp_sobelx = cv2.warpAffine(sobelx, warp_mat, (0, 0))
        warp_sobely = cv2.warpAffine(sobely, warp_mat, (0, 0))
        warp_sobelx = warp_sobelx[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        warp_sobely = warp_sobely[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # steepest image calculation
        x, y = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
        x = x.reshape(-1, 1)
        y = y.reshape(-1, 1)

        steepest_descent_image = np.zeros((tmp.shape[1]*tmp.shape[0], 6))

        for i in range(0, len(x)):
            jacobian = np.array([[x[i][0], 0, y[i][0], 0, 1, 0], [0, x[i][0], 0, y[i][0], 0, 1]])
            delta_I = np.array([warp_sobelx[y[i][0]][x[i][0]], warp_sobely[y[i][0]][x[i][0]]])

            steepest_descent_image[i] = np.dot(delta_I, jacobian).reshape(1, -1)

        # hessian calculation
        hessian = np.matmul(steepest_descent_image.T, steepest_descent_image)

        # steepest descent parameters
        sd_param = np.matmul(steepest_descent_image.T, diff)

        # change in warping parameters
        delta_p = np.matmul(np.linalg.inv(hessian), sd_param)
        delta_p *= 100
        norm_p = np.linalg.norm(delta_p)
        print(norm_p)

        # new parameters
        for i in range(len(pprev)):
            pprev[i] = delta_p[i] + pprev[i]

        # update warp matrix
        warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

    # display warped image
    cv2.imshow('warped image', warp_image)

    newrow = [0, 0, 1]
    matrix = np.vstack([warp_mat, newrow])

    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1, -1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1, -1).T

    new_rect = [(int(pt1[0][0]), int(pt1[1][0])), (int(pt2[0][0]), int(pt2[1][0]))]

    return pprev, new_rect


# main function
if __name__ == '__main__':

    # get images from folder
    filenames = glob.glob("Dataset/Car4/img/*.jpg")
    filenames.sort()
    images = [cv2.imread(img, 0) for img in filenames]

    # bounding box coordinates in image
    box = np.array([[67, 50], [178, 138]], dtype='int32')
    # box = np.array([[128, 52], [189, 139], [67, 138]], dtype='float32')

    # get template from folder
    template = cv2.imread('Dataset/Car4/img/0001.jpg', 0)
    template = template[box[0][1]:box[1][1] , box[0][0]:box[1][0]]
    # cv2.imshow('template', template)
    # cv2.waitKey(0)

    # warping parameters
    param = np.zeros(6)

    for i in range(len(images)):

        img = images[i]
        param, new_box = affineLKtracker(images[i], template, box, param)

        # display final output
        cv2.rectangle(img, new_box[0], new_box[1], (0, 0, 255), 2)
        cv2.imshow('image', img)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
