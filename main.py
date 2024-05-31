# This is a sample Python script.

import cv2
import numpy as np
from matplotlib import pyplot as plt


print("debug")

def is_rectangle(approx, angle_threshold=10, size_threshold=5000):
    """
    Check if a contour approximates a rectangle.

    Parameters:
    approx : The coordinates of the polygonal contours.
    angle_threshold : The maximum deviation from 90 degrees to still consider an angle as a right angle.
    size_threshold : Minimum area size to consider the contour as a potential parking space.

    Returns:
    True if the contour is a rectangle, False otherwise.
    """
    # A rectangle has exactly four vertices
    if len(approx) == 4:
        (x, y, w, h) = cv2.boundingRect(approx)  # Get the bounding box of the contour
        area = cv2.contourArea(approx)  # Calculate the area of the contour
        # Ensure the area is large enough and the shape is not too elongated
        if area > size_threshold and 3 < w / h < 3:
            # Check each angle of the contour to be close to 90 degrees
            for i in range(4):
                angle = angle_between(approx[i], approx[(i + 1) % 4], approx[(i + 2) % 4])
                if not (90 - angle_threshold <= angle <= 90 + angle_threshold):
                    return False
            return True
    return False


def angle_between(pt1, pt2, pt3):
    """
    Calculate the angle in degrees between points at pt2 formed between vectors pt1-pt2 and pt2-pt3.

    Parameters:
    pt1, pt2, pt3 : Coordinates of the three points.

    Returns:
    Angle in degrees.
    """
    print('check')
    # Create vectors from points
    vec1 = [pt1[0][0] - pt2[0][0], pt1[0][1] - pt2[0][1]]
    vec2 = [pt3[0][0] - pt2[0][0], pt3[0][1] - pt2[0][1]]
    # Normalize vectors
    unit_vec1 = vec1 / np.linalg.norm(vec1)
    unit_vec2 = vec2 / np.linalg.norm(vec2)
    # Calculate dot product and angle
    dot_product = np.dot(unit_vec1, unit_vec2)
    angle = np.arccos(dot_product) / np.pi * 180  # Convert radians to degrees
    return angle



def find_parking_spaces(image_path):
    """
    Identify parking spaces in an image based on their shape (rectangles).

    Parameters:
    image_path : Path to the image file.
    """
    # Load image and convert to grayscale
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect edges in the image
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    # Find contours in the edge-detected image
    contours, _ = cv2.findContours(edges, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    parking_spaces = []

    for cnt in contours:
        # Approximate each contour to polygon and check if it is a rectangle
        approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)
        if is_rectangle(approx):
            parking_spaces.append(cv2.boundingRect(approx))

    # Draw rectangles on the original image for each detected parking space
    for (x, y, w, h) in parking_spaces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    """
    plt.imshow(image)
    plt.title('Parking Spaces')
    plt.show()
    """
    print('imshow now')
    cv2.imshow('Parking Spaces', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    

# Usage
find_parking_spaces('/Users/admin/Documents/Screenshots/Screenshot 2024-05-30 at 13.01.30.png')

