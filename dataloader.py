from torch.utils.data import Dataset
import os
import cv2
from gluoncv import model_zoo, data, utils

class boundingBoxClothes(Dataset):
    def __init__(self, root_dir, transforms):
        self.root_dir = root_dir
        self.annotations = []
        #replace with good and bad
        classes = ['GOOD', 'BAD']
        
        folderNames = os.listdir('downloads')
        
        for folder in os.listdir('downloads'):
            for file in os.listdir('downloads/' + folder):
                self.annotations.append([os.path.join(self.root_dir, folder, file), classes.index(folder)])
                
        self.net = model_zoo.get_model('yolo3_darknet53_voc', pretrained=True)
        self.transforms = transforms
        
        
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, index):
        imagePath = self.annotations[index][0]
        image = cv2.imread(imagePath)
        
        x, img = data.transforms.presets.yolo.load_test(imagePath, short=512)
        class_IDs, scores, bounding_boxs = self.net(x)
        
        idx = 0
        for i, val in enumerate(class_IDs[0]):
            if val == 14:
                idx = i
                bbox = bounding_boxs[0][idx]
                crop_img = img[int(bbox[1].asscalar()):int(bbox[3].asscalar()), int(bbox[0].asscalar()):int(bbox[2].asscalar())]

                shape = crop_img.shape
                if 0 not in shape:
                    break

        bbox = bounding_boxs[0][idx]
        crop_img = img[int(bbox[1].asscalar()):int(bbox[3].asscalar()), int(bbox[0].asscalar()):int(bbox[2].asscalar())]
        
        shape = crop_img.shape
        
        if 0 in shape:
            print(imagePath)
        
        try:
            crop_img = cv2.resize(crop_img, (192, 256), interpolation=cv2.INTER_AREA)
        except:
            crop_img = img[int(bbox[1].asscalar()):int(bbox[3].asscalar()), int(bbox[0].asscalar()):int(bbox[2].asscalar())]
        

        
        if self.transforms:
            crop_img = self.transforms(crop_img)
            
           
        
        label = self.annotations[index][1]
        
        return crop_img, label