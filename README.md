# Drone Surveillance Contest

### Computer Vision Tournament - [CVZone](https://www.computervision.zone/) community.

![title](/github_images/Result.PNG)
[![title](/github_images/youtube.png "Computer Vision Zone Community Tournament - Drone Surveillance Contest - Count and track the number of cars by drone passing over a junkyard")](https://youtu.be/_s2noEVmjcI)

* Tournament name:  'Drone Surveillance Contest'.
* Tournament task:   to count and track the number of cars by drone passing over a junkyard.

<p>
<br />
</p>

### Project steps:

* Create an image dataset
  1. Split the video into frames
  2. Add new images data to prevent overfitting and achieve versatile performance
  3. Perform augmentation to achieve the optimum result for a dynamic environment
  4. Data labeling 
* Training custom CNN model (YOLO Algorithm)
* Code                
   - Initialize Model
   - Cars Detection
   - Count and track cars that are only in the region of interest.
   - Draw result
  
  
#
###### Create an image dataset
>
> 
> ##### 1. Split the video into frames
> ![title](/github_images/split_video.PNG)
>
> ##### 2. Add new images data to prevent overfitting and achieve versatile performance
> ![title](/github_images/more_data.PNG)
> ![title](/github_images/more_data2.PNG)
>
> ##### 3. Perform augmentation to achieve the optimum result for a dynamic environment
> ![title](/github_images/augmentation.PNG)
>
> ##### 4. Data labeling 
> ![title](/github_images/labelimg2.PNG)
> ![title](/github_images/labelimg_aug2.PNG)
> ![title](/github_images/more_data_label2.PNG)
>

<p>
<br />
</p>

###### Training custom CNN model (YOLO Algorithm)
> Training custom model in Yolov4 tiny architecture using Google Colab (Darknet Framework).
>

###### Code
> 
> ##### Initialize Model
>  ```
>    def initialize_model(self):
>
>        with open('./model/coco.names', 'r') as f:
>        self.class_names = [cname.strip() for cname in f.readlines()]
>   
>        net = cv2.dnn.readNet('./model/Drone_Surveillance_Contest.weights', './model/Drone_Surveillance_Contest.cfg')
>        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
>        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
>        self.model = cv2.dnn_DetectionModel(net)
>        self.model.setInputParams(1 / 255, (416, 416), (0, 0, 0), swapRB=True, crop=False)   
>  ```
>  
> ##### Cars Detection
>  ```
>    def detect(self, frame):
>        self.classes, scores, boxes = self.model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)     
>  ```
>
>  ##### Count and track cars that are only in the region of interest!
>  ```
>    def region_of_interest(self, current_frame, detection_boxes):
>        cv2.putText(current_frame, "Region scanner", (10, self.up_threshold - 20),
>                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
>
>        cv2.addWeighted(current_frame, 0.7, self.blank_image, 0.3, 0, current_frame)
>        cars_in_roi = []
>        # Count cars that are only in the region of interest!
>        for box in detection_boxes:
>            _, y, _, h = box
>            if self.up_threshold < y < self.down_threshold and h > 200:
>                cars_in_roi.append(box)
>
>        # Track and count
>        objects_bbs_ids = self.update(cars_in_roi)
>
>        # Draw cars amount and my name
>        self.draw_title(current_frame)
>
>        return objects_bbs_ids 
>  ```
>
>  ##### Draw result
>  ```
>    def draw_objects(self, frame, classes, scores, boxes):
>
>        for (class_id, score, box) in zip(classes, scores, boxes):
>
>            # Box center
>            x, y = self.get_box_center(box)
>
>            xx, yy, w, h, index = box
>
>            overlay = frame.copy()
>
>            if class_id[0] == 0:
>                cv2.rectangle(overlay, (x - w//2, y), (xx + w, yy + h), COLORS[5], -1)
>
>                # Car bounding box
>                cv2.rectangle(overlay, (xx, yy), (xx + w, yy + h), COLORS[1], 2)
>                cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
>
>                # Car number
>                cv2.putText(frame, f'  {index}', (xx, int(yy + h*0.8)), cv2.FONT_HERSHEY_SIMPLEX, 1, COLORS[4], 2)
>
>        return frame        
> ```
> 

<p>
<br />
</p>

### References
>
> - [Pysource](https://pysource.com/).
>
> - [CV Zone](https://www.computervision.zone/).
>
> - [Augmented Startups](https://www.augmentedstartups.com/).
>
> - [Roboflow](https://roboflow.com/).
>
