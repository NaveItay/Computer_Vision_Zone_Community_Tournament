import math
import cv2
import numpy as np


class CarCountAndTrack:

    # region of interest
    up_threshold = 250
    down_threshold = 265
    blank_image = np.zeros((1080, 1920, 3), np.uint8)
    cv2.rectangle(blank_image, (0, up_threshold), (1920, down_threshold), (255, 255, 255), -1)

    def __init__(self):
        # Store the center positions of the objects
        self.center_points = {}
        # Keep the count of the IDs
        # each time a new object id detected, the count will increase by one
        self.id_count = 1

    def region_of_interest(self, current_frame, detection_boxes):
        print(self.id_count - 1)

        cv2.putText(current_frame, "Region scanner", (10, self.up_threshold - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

        cv2.addWeighted(current_frame, 0.7, self.blank_image, 0.3, 0, current_frame)
        cars_in_roi = []
        # Count cars that are only in the region of interest!
        for box in detection_boxes:
            _, y, _, h = box
            if self.up_threshold < y < self.down_threshold and h > 200:
                cars_in_roi.append(box)

        # Track and count
        objects_bbs_ids = self.update(cars_in_roi)

        # Draw cars amount and my name
        self.draw_title(current_frame)

        return objects_bbs_ids

    def draw_title(self, current_frame):

        cv2.putText(current_frame, str(self.id_count - 1), (1750, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 2)
        cv2.putText(current_frame, "Itay Nave", (915, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 3)

    def update(self, objects_rect):
        # Objects boxes and ids
        objects_bbs_ids = []

        # Get center point of new object\
        for rect in objects_rect:
            x, y, w, h = rect
            cx = (x + x + w) // 2
            cy = (y + y + h) // 2

            # Find out if that object was detected already
            same_object_detected = False
            for id, pt in self.center_points.items():
                dist = math.hypot(cx - pt[0], cy - pt[1])

                if dist < 100:
                    self.center_points[id] = (cx, cy)

                    objects_bbs_ids.append([x, y, w, h, id])
                    same_object_detected = True
                    break

            # New object is detected we assign the ID to that object
            if same_object_detected is False:
                self.center_points[self.id_count] = (cx, cy)
                objects_bbs_ids.append([x, y, w, h, self.id_count])
                self.id_count += 1

        # Clean the dictionary by center points to remove IDS not used anymore
        new_center_points = {}
        for obj_bb_id in objects_bbs_ids:
            _, _, _, _, object_id = obj_bb_id
            center = self.center_points[object_id]
            new_center_points[object_id] = center

        # Update dictionary with IDs not used removed
        self.center_points = new_center_points.copy()

        return objects_bbs_ids

    def get_box_center(self, box):
        x, y, w, h = box
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy
