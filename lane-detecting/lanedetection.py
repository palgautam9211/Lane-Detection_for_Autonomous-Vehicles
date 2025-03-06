import cv2
import matplotlib.pyplot as pt
import numpy as np

def make_cordinates(image,line_parameters):
    slope,intercept=line_parameters
    y1=image.shape[0]
    y2 = int(y1 * (3/5))
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1,y1,x2,y2])
def average_slope_intercept(image,lines):
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2=line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intercept=parameters[1]
        if slope<0:
            left_fit.append((slope,intercept))
        else:
            right_fit.append((slope,intercept))
    if left_fit:
      left_fit_average=np.average(left_fit, axis=0)
      left_line = make_cordinates(image, left_fit_average)
    else:
        left_line=None
    if right_fit:
      right_fit_average=np.average(right_fit, axis=0)
      right_line=make_cordinates(image,right_fit_average)
    else:
        right_line=None
    return np.array([left_line,right_line]) if left_line is not None and right_line is not None else None


def canny(image):

    #convert in gray color to get  the gradient form of image
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    #call the canny method for identifying the edges of blur image
    canny = cv2.Canny(blur, 50, 150)
    return canny

def display_lines(image,lines):
    line_image = np.zeros_like(image)
    if lines is not None:
        for line in lines:
            if line is not None:
               x1,y1,x2,y2= line.reshape(4)
               cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image


def region_of_interest(image):

    #vertices of region that we are selected
    polygons=np.array([
        [(200, 700), (1100, 700), (550, 250)]
    ])

    #mask:  simply display the part of the image that we are interested in
    #set the image as zeros(black) color
    mask=np.zeros_like(image)

    #This function fills the polygon with color(255) on the mask
    cv2.fillPoly(mask,polygons,255)

    #masked_image = cv2.bitwise_and(originalImage,maskedImage ) performs a bitwise AND operation
    # between the original image and a mask. This operation is useful for isolating
    # or highlighting specific regions of interest in an image based on the mask you’ve created
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image


width=1279
height=704
capture = cv2.VideoCapture("video1.mp4")
if not capture.isOpened():
    print("Error: Could not open video file.")
    exit()

while capture.isOpened():
    ret, frame = capture.read()
    if not ret:
        print("Could not read frame because the video is end!")
        break

    try:
        # Safeguard the resize operation
        frame = cv2.resize(frame, (width, height))
    except Exception as e:
        print(f"Resize error: {e}")
        break

    cannyedge = canny(frame)
    cropped_image = region_of_interest(cannyedge)
    lines = cv2.HoughLinesP(cropped_image, 2, np.pi / 180, 100, np.array([]), minLineLength=40, maxLineGap=4)
    if lines is not None:
       averaged_lines = average_slope_intercept(frame, lines)
       if averaged_lines is not None:  
           line_image = display_lines(frame, averaged_lines)
           combo_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
           cv2.imshow("Result", combo_image)
       else:
           cv2.imshow("Result", frame)  # If no lines are there then show the original video

    if cv2.waitKey(20) &0xFF == ord('e'):
        break
capture.release()
cv2.destroyAllWindows()