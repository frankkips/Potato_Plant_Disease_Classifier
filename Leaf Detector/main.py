# Identify objects using openCV powered by YoloV5
import cv2
import cvlib as cv

# Load the image
image = cv2.imread('./alfred.JPG')


# perform object detection
result = cv.detect_common_objects(image)

# Check Result
if result and result[1] != []:
    for obj in result[1]:
        print("This is a ",obj)
else:
    print("No objects were found")