
import cv2
import numpy as np
from cv_bridge import CvBridge
from PIL import Image

import cv2
import qi 

class NaoqiCamera():
    def __init__(self, res=640, cam='top', fps=30, colorSpace=11): # default cam top RGB at 30 fps

        self.bridge = CvBridge()

        self.session = qi.Session()
        self.session.connect("tcp://127.0.0.1:9559")

        # self.initCamerasNaoQi(res, cam)

        self.video_service = self.session.service("ALVideoDevice")
        if cam == "top":
            camera = 0
        elif cam == "bottom":
            camera = 1

        if res == "320": 
            resolution = 1
        elif res == "640":
            resolution = 2
        elif res == "1280":
            resolution = 3

        self.videosClient = self.video_service.subscribeCamera("cameras", camera, resolution, colorSpace, fps)
    
    def get_image(self, out_format='cv2'):
        naoImage = self.video_service.getImageRemote(self.videosClient)
        if out_format == 'PIL':
            # Get the image size and pixel array.
            imageWidth = naoImage[0]
            imageHeight = naoImage[1]
            array = naoImage[6]
            image_string = str(bytearray(array))
            # Create a PIL Image from our pixel array.
            image = Image.fromstring("RGB", (imageWidth, imageHeight), image_string)
        elif out_format == 'cv2':
            image = np.frombuffer(naoImage[6], np.uint8).reshape(naoImage[1], naoImage[0], 3)
        return image

# Cameras fps resolution

# Resolution            ID Value    local	Gb Ethernet	100Mb Ethernet	WiFi g
# 40x30     (QQQQVGA)	    8       30fps	        30fps       30fps	30fps
# 80x60     (QQQVGA)	    7       30fps	        30fps	    30fps	30fps
# 160x120   (QQVGA)	        0       30fps	        30fps	    30fps	30fps
# 320x240   (QVGA)	        1       30fps	        30fps	    30fps	11fps
# 640x480   (VGA)	        2       30fps	        30fps	    12fps	2.5fps
# 1280x960  (4VGA)	        3       29fps	        10fps	    3fps	0.5fps

# Pepper camera indexes

# Parameter ID Name	    ID Value	    Type
# AL::kTopCamera	        0	    2D Camera
# AL::kBottomCamera	        1       2D Camera
# AL::kDepthCamera	        2	    Reconstructed 3D Sensor
# AL::kStereoCamera	        3       Stereo Camera


# Colorspace: 2D cameras only

# Parameter ID Name	ID Value	Number of layers	Number of channels	Description
# AL::kYuvColorSpace	0	1	1	Buffer only contains the Y (luma component) equivalent to one unsigned char
# AL::kyUvColorSpace	1	1	1	Buffer only contains the U (Chrominance component) equivalent to one unsigned char
# AL::kyuVColorSpace	2	1	1	Buffer only contains the V (Chrominance component) equivalent to one unsigned char
# AL::kRgbColorSpace	3	1	1	Buffer only contains the R (Red component) equivalent to one unsigned char
# AL::krGbColorSpace	4	1	1	Buffer only contains the G (Green component) equivalent to one unsigned char
# AL::krgBColorSpace	5	1	1	Buffer only contains the B (Blue component) equivalent to one unsigned char
# AL::kHsyColorSpace	6	1	1	Buffer only contains the H (Hue component) equivalent to one unsigned char
# AL::khSyColorSpace	7	1	1	Buffer only contains the S (Saturation component) equivalent to one unsigned char
# AL::khsYColorSpace	8	1	1	Buffer only contains the Y (Brightness component) equivalent to one unsigned char
# AL::kYUV422ColorSpace	9	2	2	Native format, 0xY’Y’VVYYUU equivalent to four unsigned char for two pixels. With Y luma for pixel n, Y’ luma for pixel n+1, and U and V are the average chrominance value of both pixels.
# AL::kYUVColorSpace	10	3	3	Buffer contains triplet on the format 0xVVUUYY, equivalent to three unsigned char
# AL::kRGBColorSpace	11	3	3	Buffer contains triplet on the format 0xBBGGRR, equivalent to three unsigned char
# AL::kHSYColorSpace	12	3	3	Buffer contains triplet on the format 0xYYSSHH, equivalent to three unsigned char
# AL::kBGRColorSpace	13	3	3	Buffer contains triplet on the format 0xRRGGBB, equivalent to three unsigned char
# AL::kYYCbCrColorSpace	14	2	2	TIFF format, four unsigned characters for two pixels.
# AL::kH2RGBColorSpace	15	3	3	H from “HSY to RGB” in fake colors.
# AL::kHSMixedColorSpace	16	3	3	HS and (H+S)/2.