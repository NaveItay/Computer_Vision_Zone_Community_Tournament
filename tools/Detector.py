import cv2

CONFIDENCE_THRESHOLD = 0.75
NMS_THRESHOLD = 0.1
COLORS = [(255, 0, 0), (0, 0, 255), (195, 195, 195), (0, 255, 0), (0, 0, 0), (255, 255, 255)]


class ObjectDetection:
    classes = []
    class_names = []
    scores = []
    boxes = []
    model = None

    def initialize_model(self):

        with open('./model/coco.names', 'r') as f:
            self.class_names = [cname.strip() for cname in f.readlines()]

        net = cv2.dnn.readNet('./model/Drone_Surveillance_Contest.weights', './model/Drone_Surveillance_Contest.cfg')
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        self.model = cv2.dnn_DetectionModel(net)
        self.model.setInputParams(1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)

    def detect(self, frame):
        self.classes, scores, boxes = self.model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)

        # Draw all cars
        self.draw_all(frame, boxes)

        return self.classes, scores, boxes

    def draw_objects(self, frame, classes, scores, boxes):

        for (class_id, score, box) in zip(classes, scores, boxes):

            # Box center
            x, y = self.get_box_center(box)

            xx, yy, w, h, index = box

            # label = "(%d, %d)" % (x, y)
            overlay = frame.copy()

            if class_id[0] == 0:
                # cv2.putText(frame, label + " " + str(score), (box[0], box[1] - 10),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[4], 2)
                # cv2.putText(frame, str(self.class_names[int(class_id)]), (x, y),
                #             cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[1], 2)
                cv2.rectangle(overlay, (x - w//2, y), (xx + w, yy + h), COLORS[5], -1)

                # Car bounding box
                cv2.rectangle(overlay, (xx, yy), (xx + w, yy + h), COLORS[1], 2)
                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)

                # Car number
                cv2.putText(frame, f'  {index}', (xx, int(yy + h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[4], 2)

        return frame

    def get_box_center(self, box):
        x, y, w, h, _ = box
        cx = x + w // 2
        cy = y + h // 2
        return cx, cy

    def draw_all(self, frame, boxes):
        overlay = frame.copy()

        for box in boxes:
            x, y, w, h = box
            cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
            cv2.rectangle(overlay, (x, y), (x + w, y + h), COLORS[5], 2)
