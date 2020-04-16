import cv2
import numpy as np

if __name__ == '__main__':
    template = cv2.imread("Dataset/templates/croppedbaby.png", 0)
    frame = cv2.imread("Dataset/DragonBaby/DragonBaby/img/0002.jpg", 0)
    # pt1 = ([0,0],[82,0],[82,67],[0,67])
    # pt2 = ([156,73],[223,73],[223,154],[156,155])
    # h, mask = cv2.findHomography(pt1,pt2,cv2.RANSAC,5.0)
    dst = np.array([[0, 0], [82, 0], [82, 67], [0, 67]], dtype="float32")
    rect = np.array([[156, 73], [223, 73], [223, 154], [156, 155]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(frame, M, (67, 82))
    difference = cv2.subtract(template, warped)
    # print(difference)
    cv2.imshow("temp", difference)
    # cv2.imshow("frame", frame)
    # cv2.imshow("warp", warped)
    cv2.waitKey()
    # print(h)