import cv2
import numpy as np
from sklearn.cluster import KMeans

# Load the foreground video and the background video
video = cv2.VideoCapture("test1.mp4")
background_video = cv2.VideoCapture("backv.mp4")

def nothing(x):
    pass

def find_hsv_bounds(frame_hsv, k=3):
    """Use K-Means clustering to find the HSV bounds for the green screen."""
    # Reshape the image into a 2D array of HSV values
    pixels = frame_hsv.reshape((-1, 3))
    
    # Apply K-Means clustering to classify the pixels
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(pixels)
    
    # Find the cluster that corresponds to green (usually the one with the highest mean Hue value)
    cluster_centers = kmeans.cluster_centers_
    
    # Sort clusters by Hue (H) to find the green cluster
    cluster_centers = sorted(cluster_centers, key=lambda x: x[0])  # Sorting by Hue (H)
    
    # Assuming the green is in the middle or lowest cluster
    green_cluster = cluster_centers[1]  # Assuming middle cluster (can adjust based on test results)
    
    # Create some margin for HSV range (increase the tolerance)
    lower_bound = np.array([max(0, green_cluster[0] - 30), max(0, green_cluster[1] - 80), max(0, green_cluster[2] - 80)])
    upper_bound = np.array([min(179, green_cluster[0] + 30), min(255, green_cluster[1] + 80), min(255, green_cluster[2] + 80)])
    
    return lower_bound, upper_bound

# Create trackbars for manual adjustment (optional)
cv2.namedWindow("Trackbars")
cv2.resizeWindow("Trackbars", 300, 300)
cv2.createTrackbar('Lower Hue', 'Trackbars', 40, 179, nothing)
cv2.createTrackbar('Upper Hue', 'Trackbars', 80, 179, nothing)
cv2.createTrackbar('Lower Saturation', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper Saturation', 'Trackbars', 255, 255, nothing)
cv2.createTrackbar('Lower Value', 'Trackbars', 50, 255, nothing)
cv2.createTrackbar('Upper Value', 'Trackbars', 255, 255, nothing)

while True:
    # Read the next frame from the foreground video
    ret, frame = video.read()
    if not ret:
        print("Foreground video ended or failed to read frame.")
        break

    # Read the next frame from the background video
    ret_bg, bg_frame = background_video.read()
    if not ret_bg:
        print("Background video ended or failed to read frame. Looping...")
        background_video.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret_bg, bg_frame = background_video.read()
        if not ret_bg:
            print("Failed to read background video after looping.")
            break

    # Resize both frames to the same size
    frame = cv2.resize(frame, (640, 480))
    bg_frame = cv2.resize(bg_frame, (640, 480)) if bg_frame is not None else np.zeros((480, 640, 3), dtype=np.uint8)

    # Convert the video frame to the HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Get manual trackbar positions (useful for real-time adjustment)
    l_h = cv2.getTrackbarPos('Lower Hue', 'Trackbars')
    u_h = cv2.getTrackbarPos('Upper Hue', 'Trackbars')
    l_s = cv2.getTrackbarPos('Lower Saturation', 'Trackbars')
    u_s = cv2.getTrackbarPos('Upper Saturation', 'Trackbars')
    l_v = cv2.getTrackbarPos('Lower Value', 'Trackbars')
    u_v = cv2.getTrackbarPos('Upper Value', 'Trackbars')

    # Set HSV bounds from trackbars (can also use find_hsv_bounds() dynamically)
    lower_bound = np.array([l_h, l_s, l_v])
    upper_bound = np.array([u_h, u_s, u_v])

    # Create a mask for the green color based on the auto-detected or manual bounds
    mask = cv2.inRange(hsv, lower_bound, upper_bound)

    # Use morphological operations to clean the mask
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)  # Close small gaps in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)   # Remove noise

    # Invert the mask to extract the non-green regions of the frame
    mask_inv = cv2.bitwise_not(mask)

    # Extract the foreground where the mask is not green
    fg_frame = cv2.bitwise_and(frame, frame, mask=mask_inv)
    
    # Extract the background where the mask is green
    bg_res = cv2.bitwise_and(bg_frame, bg_frame, mask=mask)
    
    # Combine the foreground and background
    green_screen = cv2.add(fg_frame, bg_res)

    # Display the resulting green screen effect
    cv2.imshow("Green screen", green_screen)
    # cv2.imshow("Mask", mask)
    
    # Exit when 'q' is pressed
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release video capture objects and close any OpenCV windows
video.release()
background_video.release()
cv2.destroyAllWindows()
