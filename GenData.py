# This file is used for training character recognition and the output of this file is classifications.txt and flattened_images.txt

import sys
import numpy as np
import cv2
import os

MIN_CONTOUR_AREA = 100

RESIZED_IMAGE_WIDTH = 20
RESIZED_IMAGE_HEIGHT = 30


def main():
    imgTrainingNumbers = cv2.imread("training_chars.png")            # read the training characters image

    if imgTrainingNumbers is None:                          # if image was not read successfully
        print("error: image not read from file \n\n")
        os.system("pause")
        return
    # end if

    imgGray = cv2.cvtColor(imgTrainingNumbers, cv2.COLOR_BGR2GRAY)          # get grayscale image
    imgBlurred = cv2.GaussianBlur(imgGray, (5,5), 0)                        # blur

    # filter image from grayscale to black and white
    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)

    cv2.imshow("imgThresh", imgThresh)

    imgThreshCopy = imgThresh.copy()

    npaContours, npaHierarchy = cv2.findContours(imgThreshCopy,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

                                # declare empty numpy array, we will use this to write to file later
                                # zero rows, enough cols to hold all image data
    npaFlattenedImages =  np.empty((0, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))

    intClassifications = []         # declare empty classifications list, this will be our list of how we are classifying our chars from user input, we will write to file at the end

                                    # possible chars we are interested in are digits 0 through 9, put these in list intValidChars
    intValidChars = [ord('0'), ord('1'), ord('2'), ord('3'), ord('4'), ord('5'), ord('6'), ord('7'), ord('8'), ord('9'),
                     ord('A'), ord('B'), ord('C'), ord('D'), ord('E'), ord('F'), ord('G'), ord('H'), ord('I'), ord('J'),
                     ord('K'), ord('L'), ord('M'), ord('N'), ord('O'), ord('P'), ord('Q'), ord('R'), ord('S'), ord('T'),
                     ord('U'), ord('V'), ord('W'), ord('X'), ord('Y'), ord('Z')]

    for npaContour in npaContours:                          # for each contour
        if cv2.contourArea(npaContour) > MIN_CONTOUR_AREA:          # if contour is big enough to consider
            [intX, intY, intW, intH] = cv2.boundingRect(npaContour)         # get and break out bounding rect

                                                # draw rectangle around each contour as we ask user for input
            cv2.rectangle(imgTrainingNumbers,(intX, intY),(intX+intW,intY+intH),(0, 0, 255),2)

            imgROI = imgThresh[intY:intY+intH, intX:intX+intW]                                  # crop char out of threshold image
            imgROIResized = cv2.resize(imgROI, (RESIZED_IMAGE_WIDTH, RESIZED_IMAGE_HEIGHT))     # resize image, this will be more consistent for recognition and storage

            cv2.imshow("imgROI", imgROI)                    # show cropped out char for reference
            cv2.imshow("imgROIResized", imgROIResized)      # show resized image for reference
            cv2.imshow("training_numbers.png", imgTrainingNumbers)      # show training numbers image, this will now have red rectangles drawn on it

            intChar = cv2.waitKey(0)                     # get key press

            if intChar == 27:                   # if esc key was pressed
                sys.exit()                      # exit program
            elif intChar in intValidChars:      # else if the char is in the list of chars we are looking for . . .

                intClassifications.append(intChar)                                                # append classification char to integer list of chars (we will convert to float later before writing to file)

                npaFlattenedImage = imgROIResized.reshape((1, RESIZED_IMAGE_WIDTH * RESIZED_IMAGE_HEIGHT))
                npaFlattenedImages = np.append(npaFlattenedImages, npaFlattenedImage, 0)


    fltClassifications = np.array(intClassifications, np.float32)                   # convert classifications list of ints to numpy array of floats

    npaClassifications = fltClassifications.reshape((fltClassifications.size, 1))   # flatten numpy array of floats to 1d so we can write to file later
    print("\n\ntraining complete !!\n")

    np.savetxt("classifications.txt", npaClassifications)           # write classifications
    np.savetxt("flattened_images.txt", npaFlattenedImages)          # write flattened images

    cv2.destroyAllWindows()             # remove windows from memory

    return

###################################################################################################
if __name__ == "__main__":
    main()
# end if




