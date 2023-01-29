import cv2
import numpy as np

# Create video capture object
cap = cv2.VideoCapture("freeway1.mp4")
ker = None
background = cv2.createBackgroundSubtractorMOG2(detectShadows = True)

while True:
    # Analyze frame by frame
    _, first_frame = cap.read()
    _, last_frame = cap.read()

    surface_mask = background.apply(first_frame)
    _, surface_mask = cv2.threshold(surface_mask, 250, 255, cv2.THRESH_BINARY)
    surface_mask = cv2.erode(surface_mask, ker, iterations = 1)
    surface_mask = cv2.dilate(surface_mask, ker, iterations = 2)
    
    # Compute the absolute difference between two frames
    diff = cv2.absdiff(first_frame, last_frame)

    # Convert the frame to grayscale
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)

    # Apply blurness to smoothen the frame
    diff_blur = cv2.GaussianBlur(diff_gray, (5, 5), 0)

    # Apply a threshold to highlight the moving pixels
    _, threshold_bin = cv2.threshold(diff_blur, 20, 255, cv2.THRESH_BINARY)

    # Find contours in the threshold image
    contours, hierarchy = cv2.findContours(threshold_bin, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Draw the rectangle box
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if cv2.contourArea(cnt) > 300:
            cv2.rectangle(first_frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Put text on each grounding box
            cv2.putText(first_frame, 'Motion Detected', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2, cv2.LINE_AA)
    
    surface_masked = cv2.bitwise_and(first_frame, first_frame, mask=surface_mask)
    # Combine the detected  and the mask frame
    all_frame = np.hstack((first_frame, surface_masked))

    # Display the output frames and compare them side by side
    cv2.imshow("Motion Detector and Mask", cv2.resize(all_frame, None, fx=0.5, fy=0.5))
    
    #Press 'ESC' key to exit
    if cv2.waitKey(10) == 27:
        break

cap.release()
cv2.destroyAllWindows() 