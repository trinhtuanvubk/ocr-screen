import os
import cv2
import argparse
import pandas as pd

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--recording_path", type=str, default="sample.webm")
    parser.add_argument("--method", type=str, default="paddleocr")
    parser.add_argument("--step", type=int, default=400, help="frame step")
    args = parser.parse_args()
    
    return args

def ocr(image, paddleocr):
    result = paddleocr.ocr(image, cls=True)
    result = result[0]
    boxes = [line[0] for line in result]
    txts = [line[1][0] for line in result]
    
    return boxes, txts


def pipeline(args):
    
    # Init method
    if args.method=="paddleocr":
        from paddleocr import PaddleOCR
        paddleocr = PaddleOCR(use_angle_cls=True, lang='en')
        
    # Init df
    df = pd.DataFrame(columns=['Frame_ID', 'Boxes', 'Txts'])
    
    # Read frames
    video_path = args.recording_path
    
    # Create frames folder
    video_name = video_path.rsplit("/",1)[-1].rsplit(".",1)[-1]
    frames_folder = os.path.join("frames", f"{video_name}")
    os.makedirs(frames_folder, exist_ok=True)
    
    video_capture = cv2.VideoCapture(video_path)
    # Check if the video file was successfully opened
    if not video_capture.isOpened():
        print("Error: Could not open video file.")
        exit()
    frame_count=0
    while True:
        # Read a frame from the video
        ret, frame = video_capture.read()
        # If there are no more frames to read, break out of the loop
        if not ret:
            break
        # Display the frame
        # cv2.imshow('Frame', frame)

        # Increment frame count
        frame_count+=1
        # Check step
        if frame_count % args.step==0:
            print(f"======Processing frame {frame_count}======")
            # Save the frames
            frame_path = os.path.join(frames_folder, f"{frame_count}.jpg")
            cv2.imwrite(frame_path, frame)
            
            # Get boxes, txts
            boxes, txts = ocr(frame, paddleocr)
            print(txts)
            # Write to df
            new_row = {"Frame_ID": frame_count, "Boxes":boxes, "Txts":txts}
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        
            
        # Wait for 25 milliseconds. You can adjust this value to change the playback speed.
        # Press 'q' to exit the loop
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    # Save DataFrame to Excel
    excel_path = os.path.join("results_excel", f"{video_name}.xlsx")
    df.to_excel(excel_path, index=False)
    # Release the VideoCapture object and close all windows
    video_capture.release()
    cv2.destroyAllWindows()


if __name__=="__main__":
    args = get_args()
    pipeline(args)
    
    
        
