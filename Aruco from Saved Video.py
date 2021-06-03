import cv2
import cv2.aruco as aruco
import numpy as np
import os

def loadArucoMarkers(path): # to find all the information about the markers
    print("path:", path)
    myList = os.listdir(path)  # complete list of our images
    markDics = []
    for name in myList:
        key = int(os.path.splitext(name)[0])
        markDics.append(key)
    return markDics

def findArucoMarkers(frame, markerSize = "APRILTAG", totalMarkers = "16h5", draw = True): #size of marker is 6x6 and 250 total markers; draw to flag markers
    """
    :param frame: frame in which to find the aruco markers
    :param markerSize: size of the markers (will use later for a more generalized use)
    :param totalMarkers: total number of markers that compose the dictionary (^)
    :param draw: flag to draw bbox around markers detected
    :return: bounding boxes and id numbers of markers detected
    """
    imgGray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # change frame to greyscale
    #key = getattr(aruco, f'DICT_{markerSize}_{totalMarkers}') # this will allow for a more generalized use
    arucoDict = aruco.Dictionary_get(aruco.DICT_4X4_100)    # substitute () with key when generalizing
    arucoParam = aruco.DetectorParameters_create()
    bboxs, ids, rejected = aruco.detectMarkers(imgGray, arucoDict, parameters = arucoParam)

    if draw:
        aruco.drawDetectedMarkers(frame, bboxs)   # bounding box around the marker

    return [bboxs, ids]

def main():
    cap = cv2.VideoCapture('/Users/gingasato1/PycharmProjects/Arucomarkerdetection/venv/2x_4x4_100.mov') # play video from file
    print("file exists?", os.path.exists('/Users/gingasato1/PycharmProjects/Arucomarkerdetection/Markers'))
    arucoMark = loadArucoMarkers('/Users/gingasato1/PycharmProjects/Arucomarkerdetection/Markers')
    framenum = 0
    counter = 0

    ret, frame = cap.read()
    while ret != False:
        if framenum % 3 == 0:  # looks through every 3 frames
            arucoFound = findArucoMarkers(frame)    # will call this function to detect some markers
            if len(arucoFound[0]) != 0:     # if bbox is not empty
                for id in np.unique(arucoFound[1]):  # it will look at unique aruco ids
                    if int(id) in arucoMark:
                        counter = counter + 1   # it will increase the counter if aruco id is found
                if counter >= 2:
                    cv2.imwrite(f'/Users/gingasato1/PycharmProjects/Arucomarkerdetection/framelist/2aruco_@{framenum}.png', frame)
                    print("success at frame:", framenum)
                    counter = 0
                else:
                    counter = 0
        framenum = framenum + 1
        ret, frame = cap.read()
    cap.release()

if __name__ == "__main__":
    main()