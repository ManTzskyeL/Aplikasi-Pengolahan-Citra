
import cv2
import numpy as np

def to_grayscale(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

def to_binary(img, threshold=127):
    gray = to_grayscale(img)
    _, binary = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    return binary

def add_images(img1, img2):
    return cv2.add(img1, img2)

def subtract_images(img1, img2):
    return cv2.subtract(img1, img2)

def logical_and(img1, img2):
    return cv2.bitwise_and(img1, img2)

def logical_or(img1, img2):
    return cv2.bitwise_or(img1, img2)

def logical_not(img):
    return cv2.bitwise_not(img)

def histogram(img):
    if len(img.shape) == 3:
        img = to_grayscale(img)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    return hist

def apply_filter(img, filter_type="sharpen"):
    kernels = {
        "sharpen": np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]]),
        "blur": np.ones((5,5), np.float32)/25,
        "edge": np.array([[-1,-1,-1], [-1,8,-1], [-1,-1,-1]])
    }
    kernel = kernels.get(filter_type, kernels["sharpen"])
    return cv2.filter2D(img, -1, kernel)

def morphology(img, operation="dilate", shape="rect", ksize=3):
    shapes = {
        "rect": cv2.MORPH_RECT,
        "ellipse": cv2.MORPH_ELLIPSE
    }
    kernel = cv2.getStructuringElement(shapes[shape], (ksize, ksize))
    if operation == "dilate":
        return cv2.dilate(img, kernel)
    elif operation == "erode":
        return cv2.erode(img, kernel)
    return img
