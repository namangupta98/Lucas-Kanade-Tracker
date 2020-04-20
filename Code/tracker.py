import cv2
import glob
import numpy as np
import copy
<<<<<<< HEAD

# function for gamma correction of images
def gammaCorrection(images):

    gamma = 0.5
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    res = cv2.LUT(images, lookUpTable)
    return res

def img2video(images,videopath):

    imgArray = []
    for img in images:
        height, width = img.shape
        size = (width, height)
        imgArray.append(img)
    out = cv2.VideoWriter(videopath, cv2.VideoWriter_fourcc(*'DIVX'), 30, size)
    for i in range(len(imgArray)):
        out.write(imgArray[i])
    out.release()
    return out


# function for m estimator
def mEstimator(delp, image, tmp):
    oldWarpmat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])
    newWarpmat = np.array([[1 + delp[0], delp[2], delp[4]], [delp[1], 1 + delp[3], delp[5]]])
    newwarpImg = cv2.warpAffine(image, oldWarpmat, (0, 0))
    newwarpTemp = cv2.warpAffine(tmp, newWarpmat, (0, 0))
    Lroot = cv2.subtract(newwarpTemp - newwarpImg).reshape(-1,1)
    L = residual*np.sqrt(Lroot)


=======
>>>>>>> 6ad7e6537ac5896ef8634f0861291348c29a8e37


# function to compute new warping parameters
def affineLKtracker(image, tmp, rect, pprev):

    # get warping parameters
    warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

    norm_p = 1

    # steepest image calculation
    # x = np.arange(0, tmp.shape[1], 1)
    # y = np.arange(0, tmp.shape[0], 1)
    x, y = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    steepest_descent_image = np.zeros((tmp.shape[1] * tmp.shape[0], 6))

    # gradient x and y
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    ctr = 0

<<<<<<< HEAD
    while norm_p > 0.009:

        # get warped image
        warp_image = cv2.warpAffine(image, warp_mat, (0, 0))
        warp_image = warp_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # difference of template and warped image
        diff = cv2.subtract(tmp, warp_image).reshape(-1, 1)

        # get sobel warped image
        warp_sobelx = cv2.warpAffine(sobelx, warp_mat, (0, 0))
        warp_sobely = cv2.warpAffine(sobely, warp_mat, (0, 0))
        warp_sobelx = warp_sobelx[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        warp_sobely = warp_sobely[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # for steepest descent calc
        for i in range(len(y)):
            delta_I = np.array([warp_sobelx[y[i][0]][x[i][0]], warp_sobely[y[i][0]][x[i][0]]])
            jacobian = np.array([[y[i][0], 0, x[i][0], 0, 1, 0], [0, y[i][0], 0, x[i][0], 0, 1]])
            steepest_descent_image[i] = delta_I @ jacobian

        # hessian calculation
        hessian = steepest_descent_image.T @ steepest_descent_image

        # steepest descent parameters
        sd_param = steepest_descent_image.T @ diff

        # change in warping parameters
        delta_p = np.linalg.inv(hessian) @ sd_param
        # print(delta_p)
        norm_p = np.linalg.norm(delta_p)
        delta_p *= 100
        print(norm_p)

=======
    while norm_p > 0.03:

        # get warped image
        warp_image = cv2.warpAffine(image, warp_mat, (0, 0))
        warp_image = warp_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # difference of template and warped image
        diff = cv2.subtract(tmp, warp_image).reshape(-1, 1)

        # get sobel warped image
        warp_sobelx = cv2.warpAffine(sobelx, warp_mat, (0, 0))
        warp_sobely = cv2.warpAffine(sobely, warp_mat, (0, 0))
        warp_sobelx = warp_sobelx[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        warp_sobely = warp_sobely[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # for steepest descent calc
        for i in range(len(y)):
            delta_I = np.array([warp_sobelx[y[i][0]][x[i][0]], warp_sobely[y[i][0]][x[i][0]]])
            jacobian = np.array([[y[i][0], 0, x[i][0], 0, 1, 0], [0, y[i][0], 0, x[i][0], 0, 1]])
            steepest_descent_image[i] = delta_I @ jacobian

        # hessian calculation
        hessian = steepest_descent_image.T @ steepest_descent_image

        # steepest descent parameters
        sd_param = steepest_descent_image.T @ diff

        # change in warping parameters
        delta_p = np.linalg.inv(hessian) @ sd_param
        # print(delta_p)
        norm_p = np.linalg.norm(delta_p)
        delta_p *= 100
        print(norm_p)

>>>>>>> 6ad7e6537ac5896ef8634f0861291348c29a8e37
        # new parameters
        for i in range(len(pprev)):
            pprev[i] += delta_p[i]

        # update warp matrix
        warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

        ctr += 1
        print(ctr)
        if ctr > 200:
            break

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
    filenames = glob.glob("Dataset/DragonBaby/DragonBaby/img/*.jpg")
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]

    # bounding box coordinates in image
<<<<<<< HEAD
    box = np.array([[70, 51], [177, 138]], dtype='int32')

    # get template from folder
    template = cv2.imread('Dataset/Car4/img/0001.jpg',0)
=======
    box = np.array([[160, 83], [216, 148]], dtype='int32')

    # get template from folder
    template = cv2.imread('Dataset/DragonBaby/DragonBaby/img/0001.jpg', 0)
    template = template[box[0][1]:box[1][1], box[0][0]:box[1][0]]
>>>>>>> 6ad7e6537ac5896ef8634f0861291348c29a8e37

    template = template[box[0][1]:box[1][1], box[0][0]:box[1][0]]
    correctedtemp = gammaCorrection(template)
    # warping parameters
    param = np.zeros(6)

    for im in range(len(images)):

        img = copy.deepcopy(images[im])
<<<<<<< HEAD
        corrected = gammaCorrection(img)
        # cv2.imshow("corrected", corrected)
        param, new_box = affineLKtracker(cv2.cvtColor(images[im], cv2.COLOR_BGR2GRAY)/255, correctedtemp/255, box, param)

        # display final output
        cv2.rectangle(corrected, new_box[0], new_box[1], (255, 0, 0), 2)
        cv2.imshow('image', corrected)
        img2video(corrected,'out.avi')
        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
=======
        param, new_box = affineLKtracker(cv2.cvtColor(images[im], cv2.COLOR_BGR2GRAY)/255, template/255, box, param)

        # display final output
        cv2.rectangle(img, new_box[0], new_box[1], (255, 0, 0), 2)
        cv2.imshow('image', img)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break
>>>>>>> 6ad7e6537ac5896ef8634f0861291348c29a8e37
