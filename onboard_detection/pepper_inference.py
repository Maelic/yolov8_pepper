
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

class DarkNet_YCB():
    def __init__(self):
        rospy.init_node('YoloV8', anonymous=True)

        self.bridge = CvBridge()

        self.session = qi.Session()
        self.session.connect("tcp://127.0.0.1:9559")

        self.initCamerasNaoQi()

        self.model = "models/yolov8n_ycb.quant.onnx"

        self.yolov8_detector = YOLOv8(self.model, conf_thres=0.5, iou_thres=0.5)

        self.pub_cv = rospy.Publisher(
                'yolov8_detector', Image2, queue_size=1)
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

    def normalize(self, img_pil):
        # MEAN = 255 * np.array([0.485, 0.456, 0.406])
        # STD = 255 * np.array([0.229, 0.224, 0.225])
        # x = np.array(img_pil)
        # x = x.transpose(-1, 0, 1)
        # x = (x - MEAN[:, None, None]) / STD[:, None, None]
        np_image = np.array(img_pil)
        np_image = np_image.transpose(2, 0, 1) # CxHxW
        mean_vec = np.array([0.485, 0.456, 0.406])
        std_vec = np.array([0.229, 0.224, 0.225])
        norm_img_data = np.zeros(np_image.shape).astype('float32')
        for i in range(np_image.shape[0]):
            norm_img_data[i,:,:] = (np_image[i,:,:]/255 - mean_vec[i])/std_vec[i]
                
        np_image = np.expand_dims(norm_img_data, axis=0) # 1xCxHxW

        return np_image


    def image_callback(self):
        naoImage = self.video_service.getImageRemote(self.videosClient)
        imageWidth = naoImage[0]
        imageHeight = naoImage[1]
        array = naoImage[6]
        #image_bytes = bytes(bytearray(array))
        frame = np.frombuffer(naoImage[6], np.uint8).reshape(naoImage[1], naoImage[0], 3)
        # Create a PIL Image from our pixel array.
        # im = Image.frombytes("RGB", (imageWidth, imageHeight), image_bytes)

        # newsize = (640, 480)
        # img = im.resize(newsize)
        # frame = self.normalize(img)

        time_1 = time.time()
        # Use cv_bridge() to convert the ROS image to OpenCV format

        #img = self.bridge.imgmsg_to_cv2(ros_image, "bgr8")
        
        # print(img)
        
        # Convert the image to a Numpy array since most cv2 functions
        # require Numpy arrays.
        #frame = np.array(img, dtype=np.uint8)
        # print(frame.shape)
        
  

        # Update object localizer
        boxes, scores, class_ids = self.yolov8_detector(frame)

        combined_img = self.yolov8_detector.draw_detections(frame)
        
        ros_image_yolo = self.bridge.cv2_to_imgmsg(combined_img, "rgb8")
        
        self.pub_cv.publish(ros_image_yolo)

        time_2=time.time()
        print("Detection time:", time_2 - time_1)
        print("Object detected: ", len(boxes))
       
        # combined_img = cv2.resize(combined_img,(weight,height))
        # cv2.imshow("Window detection", combined_img)
        
        # Process any keyboard commands
        # self.keystroke = cv2.waitKey(5)
        # if 32 <= self.keystroke and self.keystroke < 128:
        #     cc = chr(self.keystroke).lower()
        #     if cc == 'q':
        #         # The user has press the q key, so exit
        #         rospy.signal_shutdown("User hit q key to quit.")
        


if __name__ == '__main__':

    DarkNet_YCB()
