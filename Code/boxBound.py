import cv2
import numpy as np


def mouse_click(event, x, y, flag, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x, y)
        points.append([x, y])


if __name__ == '__main__':
    points = []
    frame = cv2.imread("dataset/DragonBaby/DragonBaby/img/0001.jpg")
    cv2.namedWindow("frame", 1)
    cv2.setMouseCallback("frame", mouse_click)

    cv2.imshow('frame', frame)
    cv2.waitKey()
    print(frame.shape)
    # frame = np.zeros((360,640,3), np.uint8)
    # print(points)
    cv2.rectangle(frame, (155,72),(222,154),(0,255,0),2)

    cv2.imshow("template", frame)
    # cv2.imwrite("dataset/templates/tempCar.png", frame)
    croppedImage = frame[72:154, 155:222]
    cv2.imwrite("dataset/templates/croppedbaby.png", croppedImage)
    # cv2.imshow("crop", croppedImage)
    cv2.waitKey()