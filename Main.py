from tools.Detector import ObjectDetection
from tools.car_counter_and_tracker import CarCountAndTrack
import cv2

o_detection = ObjectDetection()
o_detection.initialize_model()
cars_amount = CarCountAndTrack()

video_path = "Video/DRONE-SURVEILLANCE-CONTEST-VIDEO.mp4"

# Video
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    
    _, current_frame = cap.read()

    # Detect cars
    classes, scores, detection_boxes = o_detection.detect(current_frame)

    # Count and track cars that are only in the region of interest!
    objects_bbs_ids = cars_amount.region_of_interest(current_frame, detection_boxes)

    # Draw
    current_frame = o_detection.draw_objects(current_frame, classes, scores, objects_bbs_ids)

    cv2.imshow("Image", current_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# shut down capture system
cap.release()
