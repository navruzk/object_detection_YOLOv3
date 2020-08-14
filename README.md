
## Detect objects in videos using YOLOv3

This repo shows how to easily apply YOLOv3 to detect objects in videos.

### Installing

[YOLOv3](https://github.com/ayooshkathuria/pytorch-yolo-v3) is required. Users should download model's weights [file](https://pjreddie.com/media/files/yolov3.weights). It should be moved to "model" folder.

Also, OpenCV is required. 

```bash
pip install -r requirements.txt
```

## Deployment

Video should be saved in "video" folder.

```bash
python object_detection_yolov3.py
```

It displays the video and detected objects.


*Codes for YOLOv3 training on custom data will be uploaded soon*
