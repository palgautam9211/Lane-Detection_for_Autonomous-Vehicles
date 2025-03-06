import cv2
import matplotlib.pyplot as plt
import numpy as np

def canny(image):
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection
    canny = cv2.Canny(blur, 50, 150)
    return canny

# Load the image
width=1279
height=704
image_path = "image2.jpg"
image = cv2.imread(image_path)
img=cv2.resize(image,(width,height))

lane_image = np.copy(img)

# Apply Canny edge detection
canny_edges = canny(lane_image)

# Display the result
#cordinates of the lane [(300,700),(1130,700),(700,362)]
plt.imshow(canny_edges)
plt.title("Canny Edge Detection")
plt.show()
