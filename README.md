# hotornot

Developed for Hack the North 2022. This is the AI side of the project. The pipeline consists of a YOLO model that detects humans and passes the bounding boxes to a resnet that can classify outfits as fashionable or not. The YOLO model was used to limit the effects of the environment on the classification. Future work would entail using instance segmentation to further decrease the effects of the environment.
