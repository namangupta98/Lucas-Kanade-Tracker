import cv2
import glob
import numpy as np
import copy

# # function for gamma correction of images
# def gammaCorrection(images):
#
#     gamma = 1.0
#     lookUpTable = np.empty((1, 256), np.uint8)
#     for i in range(256):
#         lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
#     res = cv2.LUT(images, lookUpTable)
#     return res


# function for Huber Loss
def huber(dp, delta=2):
    L = 100000 * np.sum((delta**2) * (np.sqrt(1 + ((dp/delta) ** 2)) - 1))
    print(L)
    return L


# function for histogram equalization
def EqualizeHistogram(frame):
    new_img = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    H,S,V = cv2.split(new_img)
    clahe= cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    new_img_H = clahe.apply(H)
    new_img_S = clahe.apply(S)
    new_img_V = clahe.apply(V)
    new_img1 = cv2.merge((new_img_H,new_img_S,new_img_V))
    return cv2.cvtColor(new_img1, cv2.COLOR_HSV2BGR)


# # function for m estimator
# def mEstimator(delp, image, tmp):
#     oldWarpmat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])
#     newWarpmat = np.array([[1 + delp[0], delp[2], delp[4]], [delp[1], 1 + delp[3], delp[5]]])
#     newwarpImg = cv2.warpAffine(image, oldWarpmat, (0, 0))
#     newwarpTemp = cv2.warpAffine(tmp, newWarpmat, (0, 0))
#     Lroot = cv2.subtract(newwarpTemp - newwarpImg).reshape(-1,1)
#     L = residual*np.sqrt(Lroot)


# function to compute new warping parameters
def affineLKtracker(image, tmp, rect, pprev, threshold, scaling):

    # get warping parameters
    warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

    norm_p = 1

    # steepest image calculation
    x, y = np.meshgrid(range(tmp.shape[1]), range(tmp.shape[0]))
    x = x.reshape(-1, 1)
    y = y.reshape(-1, 1)
    steepest_descent_image = np.zeros((tmp.shape[1] * tmp.shape[0], 6))

    # gradient x and y
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)

    ctr = 0
    L = 1

    while norm_p >= threshold:

        # get warped image
        warp_image = cv2.warpAffine(image, warp_mat, (0, 0), flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP)
        warp_image = warp_image[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]

        # difference of template and warped image
        diff = cv2.subtract(tmp, warp_image).reshape(-1, 1)
        # diff = (tmp.astype(int) - warp_image.astype(int)).reshape(-1, 1)  # computing T(x)- I(w(x,p))

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
        # try:
        delta_p = np.linalg.inv(hessian) @ sd_param
        # except:
        #     pass

        # print(delta_p)
        norm_p = np.linalg.norm(delta_p)
        delta_p *= scaling
        print(norm_p)

        # call for huber loss
        # L = huber(delta_p)

        # new parameters
        for i in range(len(pprev)):
            pprev[i] += delta_p[i]

        # update warp matrix
        warp_mat = np.array([[1 + pprev[0], pprev[2], pprev[4]], [pprev[1], 1 + pprev[3], pprev[5]]])

        ctr += 1
        print(ctr)
        if ctr > 20:
            break

    # display warped image
    # cv2.imshow('warped image', warp_image)

    # newrow = [0, 0, 1]
    matrix = np.vstack([warp_mat, [0, 0, 1]])
    pt1 = matrix @ np.array([rect[0][0], rect[0][1], 1]).reshape(1, -1).T
    pt2 = matrix @ np.array([rect[1][0], rect[1][1], 1]).reshape(1, -1).T

    new_rect = [(int(pt1[0][0]), int(pt1[1][0])), (int(pt2[0][0]), int(pt2[1][0]))]

    return pprev, new_rect


# function to call car dataset
def getCar():

    # get images from folder
    filenames = glob.glob("Dataset/Car4/img/*.jpg")
    filenames.sort()
    photos = [cv2.imread(img) for img in filenames]

    # image enhancement
    # for img in photos:
    #     phots = EqualizeHistogram(img)

    # bounding box coordinates in image
    box_coordinates = np.array([[70, 51], [177, 138]], dtype='int32')

    # get template from folder
    templ = cv2.imread('Dataset/Car4/img/0001.jpg')

    # size of image
    h, w, _ = templ.shape
    size = (w, h)

    templ = templ[box_coordinates[0][1]:box_coordinates[1][1], box_coordinates[0][0]:box_coordinates[1][0]]

    # scaling and threshold factor
    thresh = 0.008
    scale = 80

    return photos, box_coordinates, templ, thresh, scale, size


# function to call bolt dataset
def getBolt():

    # get images from folder
    filenames = glob.glob("Dataset/Bolt2/img/*.jpg")
    filenames.sort()
    photos = [cv2.imread(img) for img in filenames]

    # bounding box coordinates in image
    box_coordinates = np.array([[266, 80], [307, 143]])

    # get template from folder
    templ = cv2.imread('Dataset/Bolt2/img/0001.jpg')

    # size of image
    h, w, _ = templ.shape
    size = (w, h)

    templ = templ[box_coordinates[0][1]:box_coordinates[1][1], box_coordinates[0][0]:box_coordinates[1][0]]

    # scaling and threshold factor
    thresh = 0.05
    scale = 10

    return photos, box_coordinates, templ, thresh, scale, size


# function to call baby dataset
def getBaby():

    # get images from folder
    filenames = glob.glob("Dataset/DragonBaby/DragonBaby/img/*.jpg")
    filenames.sort()
    photos = [cv2.imread(img) for img in filenames]

    # bounding box coordinates in image
    box_coordinates = np.array([[160, 83], [216, 148]], dtype='int32')

    # get template from folder
    templ = cv2.imread('Dataset/DragonBaby/DragonBaby/img/0001.jpg')

    # size of image
    h, w, _ = templ.shape
    size = (w, h)

    templ = templ[box_coordinates[0][1]:box_coordinates[1][1], box_coordinates[0][0]:box_coordinates[1][0]]

    # scaling and threshold factor
    thresh = 0.03
    scale = 80

    return photos, box_coordinates, templ, thresh, scale, size


# main function
if __name__ == '__main__':

    # dataset selection
    choice = int(input('Select dataset to track: (1) Car \t (2) Bolt \t (3) DragonBaby :-  '))

    if choice == 1:
        # call car dataset
        images, box, template, thrshold, scaler, size = getCar()

    elif choice == 2:
        # cal bolt dataset
        images, box, template, thrshold, scaler, size = getBolt()

    elif choice == 3:
        # cal baby dataset
        images, box, template, thrshold, scaler, size = getBaby()

    else:
        print('Invalid Input!!!')
        exit()

    # gamma correction
    # correctedtemp = gammaCorrection(template)

    # warping parameters
    param = np.zeros(6)

    out = cv2.VideoWriter('out.avi', cv2.VideoWriter_fourcc(*'DIVX'), 10, size)

    frame_ctr = 0

    for frame in images:

        imge = copy.deepcopy(frame)

        # in-case of image enhancement
        frame = EqualizeHistogram(frame)
        template = EqualizeHistogram(template)

        # get updated parameters
        param, new_box = affineLKtracker(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                                         cv2.cvtColor(template, cv2.COLOR_BGR2GRAY), box, param, thrshold, scaler)

        # display final output
        cv2.rectangle(frame, new_box[0], new_box[1], (255, 0, 0), 2)
        cv2.imshow('image', frame)

        # write video
        out.write(frame)

        if cv2.waitKey(1) and 0xFF == ord('q'):
            break

    out.release()
