
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
from yolov8.utils import draw_bounding_box_opencv
from yolov8.utils import class_names as CLASSES

class DarkNet_YCB():
    def __init__(self):
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()

        self.session = qi.Session()
        self.session.connect("tcp://127.0.0.1:9559")

        self.initCamerasNaoQi()

        self.model = "models/yolov8n_ycb.onnx"

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

    def initCamerasNaoQi(self):
        self.video_service = self.session.service("ALVideoDevice")
        fps = 30
        resolution = 2  	# 2 = Image of 640*480px ; 3 = Image of 1280*960px
        colorSpace = 11  	# RGB
        self.videosClient = self.video_service.subscribeCamera("cameras", 0, resolution, colorSpace, fps)

    def initCameras(self):
        self.image_sub = message_filters.Subscriber(
            "/naoqi_driver/camera/front/image_raw", Image)
        self.ts = message_filters.ApproximateTimeSynchronizer(
            [self.image_sub], queue_size=10, slop=0.5)
        self.ts.registerCallback(self.image_callback)

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
                'class_name': CLASSES[class_ids[index]],
                'confidence': scores[index],
                'box': box,
                'scale': scale}
            detections.append(detection)
            img = draw_bounding_box_opencv(orig_image, class_ids[index], scores[index], round(box[0] * scale), round(box[1] * scale),
                            round((box[0] + box[2]) * scale), round((box[1] + box[3]) * scale))

        time_2=time.time()
        print("Detection time OPENCV:", time_2 - time_1)
        print("Object detected OPENCV: ", detections)
        return img

    def image_callback(self):
        naoImage = self.video_service.getImageRemote(self.videosClient)
        #image_bytes = bytes(bytearray(array))
        frame = np.frombuffer(naoImage[6], np.uint8).reshape(naoImage[1], naoImage[0], 3)
        # Create a PIL Image from our pixel array.
        # im = Image.frombytes("RGB", (imageWidth, imageHeight), image_bytes)
        onnx_out = self.detect_onnx(frame)
        print("##############################")
        opencv_out = self.detect_opencv(frame)

        ros_image_yolo_cv = self.bridge.cv2_to_imgmsg(opencv_out, "rgb8")

        self.pub_cv2.publish(ros_image_yolo_cv)
        
        ros_image_yolo_onnx = self.bridge.cv2_to_imgmsg(onnx_out, "rgb8")
        self.pub_cv.publish(ros_image_yolo_onnx)


if __name__ == '__main__':

    DarkNet_YCB()
