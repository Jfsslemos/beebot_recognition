import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from ultralytics import YOLO
import numpy as np
from cv_bridge import CvBridge
from PIL import Image as IMG
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchvision import transforms

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias=False), #bias = falso porque o batchnorm vai cancelar ele
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.conv(x)
class UNET(nn.Module):
    def __init__(
            self, in_channels=3, out_channels=1, features=[64, 128, 256, 512],
    ):
        super(UNET, self).__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        #fazendo a parte de baixo da arquitetura aqui
        for feature in features:  #para cada valor feature na lista uma camada é adicionada a lista
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        #parte de cima da arquitetura
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose2d(
                    feature*2, feature, kernel_size=2, stride=2,
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        self.bottleneck = DoubleConv(features[-1], features[-1]*2)
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skip_connections = skip_connections[::-1]

        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x)
            skip_connection = skip_connections[idx//2]
            #se não der pra concatenar devido ao tamanho da entrada

            if x.shape != skip_connection.shape:
                 x = TF.resize(x, size=skip_connection.shape[2:], antialias=None)   #Antialias = None porque vai atualizar e mudar o modo padrão para True então manter em None

            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip)


        return self.final_conv(x)

class BeebotRecognition(Node):
    
    def __init__(self):
        super().__init__('corrosion_recogniton_node')
        self.declare_parameters(
            namespace='',
            parameters=[
                ('subscribers.image_rgb', rclpy.Parameter.Type.STRING),
                ('publishers.debug_image', rclpy.Parameter.Type.STRING),
                ('publishers.recognition_topic', rclpy.Parameter.Type.STRING),
                ('network.model', rclpy.Parameter.Type.STRING),
                ('network.weight_path', rclpy.Parameter.Type.STRING)
            ])
        
        self.readParams()

        self.createComms()

        self.loadModel()

        self.loadCVBrige()
        
    
    def callback(self, msg):
        if self.network_model == 'yolov8':
            img = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            results = self.model(img)
            for r in results:
                im_array = r.plot()
                im = IMG.fromarray(im_array[..., ::-1])
            img_writen = self.cv_bridge.cv2_to_imgmsg(np.array(im), encoding='rgb8')
            self.publisher_.publish(img_writen)

        elif self.network_model == "unet2":
            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0,0,0], std=[1,1,1]),
            ])
            img = self.cv_bridge.imgmsg_to_cv2(msg,desired_encoding='bgr8')
            image = preprocess(img)
            image = image.unsqueeze(0)

            with torch.no_grad():
                output = torch.sigmoid(self.model(image))
                output = (output > 0.5).float()
            img_writen = self.cv_bridge.cv2_to_imgmsg(np.array(output), encoding='rgb8')
            self.publisher_.publish(img_writen)
            
    def loadCVBrige(self):
        self.cv_bridge = CvBridge()

    def createComms(self):
        self.publisher_ = self.create_publisher(Image, self.debug_image_topic, 1)
        self.subscription = self.create_subscription(Image, self.image_rgb, self.callback, qos_profile=1)
        self.subscription
    
    def readParams(self):
        self.debug_image_topic = self.get_parameter('publishers.debug_image').value
        self.image_rgb = self.get_parameter('subscribers.image_rgb').value
        self.recognition_topic = self.get_parameter('publishers.recognition_topic').value
        self.network_model = self.get_parameter('network.model').value
        self.network_weight_path = self.get_parameter('network.weight').value
        self.get_logger().info(f"Received: {self.debug_image_topic}")
        self.get_logger().info(f"Received: {self.image_rgb}")
        self.get_logger().info(f"Received: {self.recognition_topic}")
        self.get_logger().info(f"Received: {self.network_model}")
        self.get_logger().info(f"Received: {self.network_weight_path}")

    def loadModel(self):
        if self.network_model == 'yolov8':
            self.get_logger().info("=> Loading yolov8 weight")
            self.model = YOLO(self.network_weight_path)

        elif self.network_model == "unet2":
            self.get_logger().info("=> Loading unet2 weight")
            DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
            self.model = UNET(in_channels=3, out_channels=1).to(DEVICE)
            weight = torch.load(self.network_weight_path, map_location=torch.device(DEVICE))
            self.model.load_state_dict(weight["state_dict"])

def main(args=None):
    rclpy.init(args=args)
    node = BeebotRecognition()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
