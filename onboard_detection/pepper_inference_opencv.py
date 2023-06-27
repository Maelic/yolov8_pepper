
import time
import cv2
import numpy as np
from cv_bridge import CvBridge
import rospy
import message_filters
from PIL import Image
from sensor_msgs.msg import Image as Image2

import cv2
import qi 

from yolov8 import YOLOv8
# from yolov8.utils import draw_bounding_box_opencv
# from yolov8.utils import class_names as CLASSES

# class_names = ['002_master_chef_can',
# '003_cracker_box',
# '004_sugar_box',
# '005_tomato_soup_can',
# '006_mustard_bottle',
# '007_tuna_fish_can',
# '008_pudding_box',
# '009_gelatin_box',
# '010_potted_meat_can',
# '011_banana',
# '019_pitcher_base',
# '021_bleach_cleanser',
# '024_bowl',
# '025_mug',
# '035_power_drill',
# '036_wood_block',
# '037_scissors',
# '040_large_marker',
# '051_large_clamp',
# '052_extra_large_clamp',
# '061_foam_brick']

class_names = ['002_master_chef_can', '003_cracker_box', '004_sugar_box', '005_tomato_soup_can', '006_mustard_bottle', '007_tuna_fish_can', '008_pudding_box', '009_gelatin_box', '010_potted_meat_can', '011_banana', '021_bleach_cleanser', '024_bowl', '025_mug', '035_power_drill', '036_wood_block', '037_scissors', '040_large_marker', '051_large_clamp', '052_extra_large_clamp', '061_foam_brick']


import argparse
from Naoqi_camera import NaoqiCamera

class Detector():
    def __init__(self, model, res):
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()

        self.session = qi.Session()
        self.session.connect("tcp://127.0.0.1:9559")

        self.cam = NaoqiCamera(res, "top")

        self.model = model

        self.cv2_detector = cv2.dnn.readNetFromONNX(self.model)

        self.yolov8_detector = YOLOv8(self.model, conf_thres=0.5, iou_thres=0.5)

        self.pub_cv = rospy.Publisher(
                'yolov8_detector', Image2, queue_size=1)

        self.pub_cv2 = rospy.Publisher(
                'yolov8_detector_cv', Image2, queue_size=1)
        
        rospy.on_shutdown(self.cleanup)	

        # spin
        print("Waiting for image topics...")
        while not rospy.is_shutdown():
            self.image_callback()
        # rospy.spin()

    def cleanup(self):
        print("Cleaning up...")
        self.video_service.unsubscribe(self.videosClient)
        self.session.close()


    def draw_bounding_box_opencv(self, img, class_id, confidence, x, y, x_plus_w, y_plus_h):
        # Create a list of colors for each class where each color is a tuple of 3 integer values
        rng = np.random.default_rng(3)
        colors = rng.uniform(0, 255, size=(len(class_names), 3))    

        label = f'{class_names[class_id]} ({confidence:.2f})'
        color = colors[class_id]
        cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 2)
        cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        return img    
              
    def detect_onnx(self, image):
        time_1 = time.time()

        boxes, scores, class_ids = self.yolov8_detector(image)

        combined_img = self.yolov8_detector.draw_detections(image)
        time_2=time.time()
        print("Detected class ids: ", class_ids)
        print("Detection time ONNX:", time_2 - time_1)
        print("Object detected ONNX: ", len(boxes))

        return combined_img

    def detect_opencv(self, orig_image):
        time_1 = time.time()

        [height, width, _] = orig_image.shape
        length = max((height, width))
        image = np.zeros((length, length, 3), np.uint8)
        image[0:height, 0:width] = orig_image
        scale = length / 640

        blob = cv2.dnn.blobFromImage(image, scalefactor=1 / 255, size=(640, 640))
        self.cv2_detector.setInput(blob)
        outputs = self.cv2_detector.forward()

        outputs = np.array([cv2.transpose(outputs[0])])
        rows = outputs.shape[1]

        boxes = []
        scores = []
        class_ids = []

        for i in range(rows):
            classes_scores = outputs[0][i][4:]
            (minScore, maxScore, minClassLoc, (x, maxClassIndex)) = cv2.minMaxLoc(classes_scores)
            if maxScore >= 0.25:
                box = [
                    outputs[0][i][0] - (0.5 * outputs[0][i][2]), outputs[0][i][1] - (0.5 * outputs[0][i][3]),
                    outputs[0][i][2], outputs[0][i][3]]
                boxes.append(box)
                scores.append(maxScore)
                class_ids.append(maxClassIndex)

        result_boxes = cv2.dnn.NMSBoxes(boxes, scores, 0.25, 0.45, 0.5)

        detections = []
        for i in range(len(result_boxes)):
            index = result_boxes[i]
            box = boxes[index]
            detection = {
                'class_id': class_ids[index],
                'class_name': class_names[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            img = self.draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        time_2=time.time()
        print("Detection time OPENCV:", time_2 - time_1)
        print("Object detected OPENCV: ", detections)
        if len(detections) == 0:
            return orig_image
        return img

    def image_callback(self):
        frame = self.cam.get_image('cv2')

        # onnx_out = self.detect_onnx(frame)
        print("##############################")
        opencv_out = self.detect_opencv(frame)

        ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(opencv_out, "rgb8")

        self.pub_cv2.publish(ros_image_yolo_cv)
        
        # ros_image_yolo_onnx = self.bridge.cv2_to_imgmsg(onnx_out, "rgb8")
        # self.pub_cv.publish(ros_image_yolo_onnx)


if __name__ == '__main__':
    # get arg 1 and 2
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='yolov8n_ycb.onnx', help='model path')
    parser.add_argument('--res', type=str, default='640', help='resolution')

    args = parser.parse_args()
    model = args.model
    res = args.res

    Detector(model, res)
