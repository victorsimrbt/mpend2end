import cv2
import os
import blur_detector
from matplotlib import pyplot as plt
# Open a connection to the default camera (usually the webcam)
cap = cv2.VideoCapture(0)
blurs = []

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video stream or file")
    exit()
    
def composite(im1,im2):
    # ! im2 is the depthmap
    background,foreground = im1,im2
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)

    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    cv2.imshow("Composited image", background)
    cv2.waitKey(0)

# Read and display frames in a loop
cnt = 0
for i in range(32):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # If frame reading was not successful, break the loop
    if not ret:
        print("Error: Failed to capture image")
        break

    # Display the resulting frame
    flipped = cv2.flip(frame, 0)
    flipped = cv2.flip(flipped, 1)
    cv2.imshow('Camera Output', flipped)
    # microscope image is flipped for some reason

    # Press 'q' to exit the video display
    print(flipped.shape)
    gray = cv2.cvtColor(flipped, cv2.COLOR_BGR2GRAY)
    blur_map = blur_detector.detectBlur(gray, downsampling_factor=4, num_scales=4, scale_start=2, num_iterations_RF_filter=3, show_progress=True)
    blurs.append(blur_map)
    # image = cv2.cvtColor(flipped, cv2.COLOR_BGR2RGB)
    image = flipped
    fig, ax = plt.subplots()
    ax.imshow(image)
    ax.imshow(blur_map, alpha=0.5)
    ax.axis('off')

    cv2.imwrite(os.path.join(r"../captures",'image{}.jpg'.format(cnt)), flipped)
    plt.show()
    cnt += 1
    
    

# Release the camera and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
