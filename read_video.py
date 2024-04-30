import cv2
import os
# Path to the WebM file
webm_file = "sample.webm"

# Open the WebM file
video_capture = cv2.VideoCapture(webm_file)

# Check if the video file was successfully opened
if not video_capture.isOpened():
    print("Error: Could not open video file.")
    exit()

frame_count = 0

# Read and display each frame
while True:
    # Read a frame from the video
    ret, frame = video_capture.read()

    # If there are no more frames to read, break out of the loop
    if not ret:
        break

    # Display the frame
    # cv2.imshow('Frame', frame)

    # Increment frame count
    frame_count += 1
    
    if frame_count%100==0:
        cv2.imwrite(os.path.join("frames", f"{frame_count}.jpg"), frame)

    # Wait for 25 milliseconds. You can adjust this value to change the playback speed.
    # Press 'q' to exit the loop
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break
print(frame_count)
# Release the VideoCapture object and close all windows
video_capture.release()
cv2.destroyAllWindows()

print("Total frames:", frame_count)
