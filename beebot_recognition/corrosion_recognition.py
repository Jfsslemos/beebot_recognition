import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
from PIL import Image as IMG
import numpy
class BeebotRecognition(Node):

    def __init__(self):
        super().__init__('corrosion_recogniton_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('subscribers.image_rgb', rclpy.Parameter.Type.STRING),
                ('publishers.debug_image', rclpy.Parameter.Type.STRING),
                ('publishers.recognition_topic', rclpy.Parameter.Type.STRING)
            ])
        self.readParams()
        self.get_logger().info(f"Received: {self.debug_image_topic}")
        self.get_logger().info(f"Received: {self.image_rgb}")
        self.get_logger().info(f"Received: {self.recognition_topic}")
        self.publisher_ = self.create_publisher(Image, self.debug_image_topic, 1)
        self.subscription = self.create_subscription(Image, self.image_rgb, self.callback, qos_profile=1)
        self.subscription
        self.model = YOLO('yolov8n.pt')
        self.cv_bridge = CvBridge()
    
    def callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        results = self.model(img)
        for r in results:
            im_array = r.plot()
            im = IMG.fromarray(im_array[..., ::-1])
        img_writen = self.cv_bridge.cv2_to_imgmsg(numpy.array(im), encoding='rgb8')
        #self.get_logger().info(f"Received: {results}")
        self.publisher_.publish(img_writen)
    
    def readParams(self):
        self.debug_image_topic = self.get_parameter('publishers.debug_image').value
        self.image_rgb = self.get_parameter('subscribers.image_rgb').value
        self.recognition_topic = self.get_parameter('publishers.recognition_topic').value

def main(args=None):
    rclpy.init(args=args)
    node = BeebotRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
