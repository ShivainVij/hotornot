import torch
from gluoncv import model_zoo, data, utils
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18
import torchvision

def predict(imgPath):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load('model.pth')
    net = torch.load('net.pth')
    model.eval()
    model.to(device)
    
    x, img = data.transforms.presets.yolo.load_test(imgPath, short=512)
    class_IDs, scores, bounding_boxs = net(x)

    idx = 0
    for i, val in enumerate(class_IDs[0]):
        if val == 14:
            idx = i
            break

    bbox = bounding_boxs[0][idx]
    crop_img = img[int(bbox[1].asscalar()):int(bbox[3].asscalar()), int(bbox[0].asscalar()):int(bbox[2].asscalar())]
    
    resized = cv2.resize(crop_img,(192, 256))
    
    converter = transforms.ToTensor()
    tensorResized = converter(resized)
    
    output= model(converter(tensorResized.numpy().transpose(2, 1, 0))[None,...].to(device))
    
    if output > 0.4:
        return "NOT"
    else:
        return "HOT"
    
    