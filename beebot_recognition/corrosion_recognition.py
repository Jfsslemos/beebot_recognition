import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
from PIL import Image as IMG
import numpy
class TestYAMLParams(Node):

    def __init__(self):
        super().__init__('your_amazing_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('bool_value', None),
                ('int_number', None),
                ('float_number', None),
                ('str_text', None),
                ('bool_array', None),
                ('int_array', None),
                ('float_array', None),
                ('str_array', None),
                ('bytes_array', None),
                ('nested_param.another_int', None)
            ])
        self.publisher_ = self.create_publisher(Image, 'debug_img', 1)
        self.subscription = self.create_subscription(Image, '/image_raw', self.callback, qos_profile=1)
        self.subscription
        self.model = YOLO('/home/joao/Downloads/best.pt')
        self.cv_bridge = CvBridge()

    def callback(self, msg):
        img = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
        results = self.model(img)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = IMG.fromarray(im_array[..., ::-1])  # RGB PIL image
        img_writen = self.cv_bridge.cv2_to_imgmsg(numpy.array(im), encoding='rgb8')
        #self.get_logger().info(f"Received: {results}")
        self.publisher_.publish(img_writen)

def main(args=None):
    rclpy.init(args=args)
    node = TestYAMLParams()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
