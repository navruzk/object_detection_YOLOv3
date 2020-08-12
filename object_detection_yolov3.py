import numpy as np
import cv2

threshold = 0.4

with open("model/YOLOv3/coco.names") as file:
    classes = file.read().strip().split("\n")
    
net = cv2.dnn.readNetFromDarknet("model/YOLOv3/yolov3.cfg", "model/YOLOv3/yolov3.weights")

cap = cv2.VideoCapture("video/video.mp4")

while cv2.waitKey(1) < 0:

    status, frame = cap.read()

    blob_from_image = cv2.dnn.blobFromImage(frame, 1 / 255, (416, 416))
    net.setInput(blob_from_image)

    net_result = net.forward(net.getUnconnectedOutLayersNames())

    class_id_list = []
    probability_list = []
    box_list = []

    ## loop through results to collect accurately detected objects
    ## l - left, r - right, t - top, b - bottom, w - width, h - height
    for results in net_result:

        for result in results:

            scores = result[5:]
            class_id = np.argmax(scores)
            probability = scores[class_id]

            if probability > threshold:

                class_id_list.append(class_id)
                probability_list.append(float(probability))

                x_center = int(result[0] * frame.shape[1])
                y_center = int(result[1] * frame.shape[0])
                w = int(result[2] * frame.shape[1])
                h = int(result[3] * frame.shape[0])
                left = x_center - w // 2
                top = y_center - h // 2

                box_list.append([left, top, w, h])

    ## remove low prob boxes
    index_list = cv2.dnn.NMSBoxes(box_list, probability_list, threshold, 0.4)

    

    for index in index_list:
        index = index[0]
        box = box_list[index]
        l = box[0]
        t = box[1]
        w = box[2]
        h = box[3]

        r = l + w
        b = t + h
        class_id = class_id_list[index]

        probability = probability_list[index]
        
        ## make blue rectangle around the object
        cv2.rectangle(frame, (r, b), (l, t), (255, 0, 0), 5)
        label = classes[class_id] + " : " + str(round(probability * 100, 3))
        font = cv2.FONT_HERSHEY_SIMPLEX
        org = (50, 50)
        fontScale = 1
        color = (0, 0, 255) ##red color label
        thickness = 2
        
        ## put object label 
        cv2.putText(
            frame, label, (r, b), font, fontScale, color, thickness, cv2.LINE_AA
        )

    ## resize the windows to fit the frame
    cv2.namedWindow("Image", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Image", 2000, 1000)

    ## show the frame
    cv2.imshow("Image", frame)
