import cv2
import numpy as np
from matplotlib import pyplot as plt

def is_rectangle(approx, angle_threshold=20, size_threshold=50):
    """
    Check if a contour approximates a rectangle based on the number of vertices
    and the angles between them.

    Parameters:
    approx : array of contour points.
    angle_threshold : the maximum deviation from 90 degrees to still consider an angle a right angle.
    size_threshold : minimum area size to consider the contour as a potential parking space.

    Returns:
    True if the contour is a rectangle, False otherwise.
    """
    min_sides = 2


    if len(approx) < min_sides:

        return False

    (x, y, w, h) = cv2.boundingRect(approx)
    #print(x,y,w,h)
    area = cv2.contourArea(approx)
    #print(area)
    if area > size_threshold:
        #print('yays')
        for i in range(4):
            angle = angle_between(approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4])
            if not ((90 - angle_threshold) <= angle <= (90 + angle_threshold)):
                #print(angle)
                return False
        print("yay", angle)
        return True


def angle_between(pt1, pt2, pt3):
    """
    Calculate the angle in degrees between points at pt2 formed between vectors pt1-pt2 and pt2-pt3.

    Parameters:
    pt1, pt2, pt3 : Coordinates of the three points.

    Returns:
    Angle in degrees.
    """
    vec1 = [pt1[0][0] - pt2[0][0], pt1[0][1] - pt2[0][1]]
    vec2 = [pt3[0][0] - pt2[0][0], pt3[0][1] - pt2[0][1]]
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product) / np.pi * 180  # Convert radians to degrees
    return angle

def find_parking_spaces(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parking_spaces = []

    for cnt in contours:
        #print(cnt)
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        #cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)
        if is_rectangle(approx):
            print ('woop')
            parking_spaces.append(cv2.boundingRect(approx))
            cv2.drawContours(image, [cnt], -1, (0, 255, 0), 2)

    for (x, y, w, h) in parking_spaces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert BGR image to RGB before displaying with Matplotlib
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    plt.imshow(image_rgb)
    plt.title('Parking Spaces')
    plt.show()
    print(len(parking_spaces))


# Example usage
find_parking_spaces('/Users/admin/Downloads/justdoparkinglot1.jpeg')
