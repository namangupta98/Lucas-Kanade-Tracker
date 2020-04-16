import cv2
import glob

if __name__ == '__main__':

    filenames = glob.glob("Dataset/DragonBaby/DragonBaby/img/*.jpg")
    filenames.sort()
    images = [cv2.imread(img) for img in filenames]

    for img in images:
        cv2.imshow('image', img)
        cv2.waitKey(0)
