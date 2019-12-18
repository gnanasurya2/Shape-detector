import cv2 as cv
import numpy as np


cap = cv.VideoCapture(0)

font = cv.FONT_HERSHEY_COMPLEX

while True:
    # cap.read() reads an frame and stores it in the variable frame
    _, frame = cap.read()
    cv.imshow("oringinal", frame)
    # this is to convert the RGB to grayscale as it will be easier to compute and there is no use of color in detecting the shape of an objects and moreover the canny algorithm will only work for grayscale images
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    cv.imshow("grayscale", gray)
    # this is to detect all the edges in the frame (https://towardsdatascience.com/canny-edge-detection-step-by-step-in-python-computer-vision-b49c3a2d8123)
    edge = cv.Canny(gray, 100, 200)
    # this is to give all the boundaries of closed objects(contours) in the image.It uses binary image to process it.Since the image form edge detection algorithm is an binary image , there is no need for converting it.
    contour, heire = cv.findContours(
        edge, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contour:
        # calculates the area of contours
        area = cv.contourArea(cnt)
        # this is avoid detecting very small objects in the frame
        if area > 200:
            # this outlines the contours in the frame with green color
            cv.drawContours(frame, [cnt], 0, (0, 255, 0), 2)
            # this is to approximate a polygon with another polygon with less vertices
            approx = cv.approxPolyDP(cnt, 0.02*cv.arcLength(cnt, True), True,)
            x = approx.ravel()[0]
            y = approx.ravel()[1]
            if len(approx) == 3:
                cv.putText(frame, "Triangle", (x, y), font, 1, (255, 0, 0))
            elif len(approx) == 4:
                cv.putText(frame, "rectangle", (x, y), font, 1, (255, 0, 0))
            elif len(approx) == 5:
                cv.putText(frame, "pentagon", (x, y), font, 1, (0, 255, 0))
            elif len(approx) == 6:
                cv.putText(frame, "hexagon", (x, y), font, 1, (0, 255, 0))
            else:
                cv.putText(frame, "circle", (x, y), font, 1, (0, 255, 0))
    cv.imshow("edge", edge)
    cv.imshow("Final", frame)
    key = cv.waitKey(1)
    if key == 27:
        break
cap.release()
cv.destroyAllWindows()
