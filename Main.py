from tools.Detector import ObjectDetection
from tools.car_counter_and_tracker import CarCountAndTrack
import cv2

o_detection = ObjectDetection()
cars_amount = CarCountAndTrack()

# Initialize Model
o_detection.initialize_model()

video_path = "Video/DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4"

cap = cv2.VideoCapture(video_path)

# Video out
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
out = cv2.VideoWriter('outpy.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), fps, (width, height))

while cap.isOpened():

    _, current_frame = cap.read()

    # Cars Detection
    classes, scores, detection_boxes = o_detection.detect(current_frame)

    # Count and track cars that are only in the region of interest!
    objects_bbs_ids = cars_amount.region_of_interest(current_frame, detection_boxes)

    # Draw result
    current_frame = o_detection.draw_objects(current_frame, classes, scores, objects_bbs_ids)

    # Write the frame into the file 'output.avi'
    out.write(current_frame)

    cv2.imshow("Image", current_frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# When everything done, release the video capture and video write objects
cap.release()
out.release()
