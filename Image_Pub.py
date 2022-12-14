#! /usr/bin/env python
from Inverse import *
from Transform import *
import cv2
import numpy as np
from math import degrees
import pandas as pd
INPUT_WIDTH=640
INPUT_HEIGHT=640

SCORE_THRESHOLD=0.5
NMS_THRESHOLD=0.45
CONFIDENCE_THRESHOLD=0.45

FONT_FACE = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE = 0.7
THICKNESS = 1

# Colors.
BLACK = (0, 0, 0)
BLUE = (255, 178, 50)
YELLOW = (0, 255, 255)

#DataFrame

#Pixel to cm conversion
width=1280
height=720
pixel_to_cmX=73/1280.0
pixel_to_cmY=40.5/720.0
classes = ['Ripe','Green','Half-Ripe']
mappings=dict()
def convertToTicks(angle):
	return round(4096/360*angle)
def pixel_to_cm(img_x,img_y):
    rw_x=pixel_to_cmX*img_x
    rw_y=pixel_to_cmY*img_y
    return rw_x,rw_y


def cam2rbt(cam_x,cam_y):
        robot_x,robot_y=transform(cam_x, cam_y)
        return (round(robot_x[0]),round(robot_y[0]))

# def findAngles(rx,ry):
#     try:
#         theta2=q2(rx,ry)
#         theta1=q1(rx,ry,theta2)
#         print (round(degrees(theta1)),round(degrees(theta2)))
#         return (round(degrees(theta1)),round(degrees(theta2)))
#     except:
#         return (-1,-1)

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED)
    # Display text inside the rectangle.
    cv2.putText(im, label, (x, y + dim[1]), FONT_FACE, FONT_SCALE, YELLOW, THICKNESS, cv2.LINE_AA)

def pre_process(input_image, net):
    # Create a 4D blob from a frame.
    blob = cv2.dnn.blobFromImage(input_image, 1 / 255, (INPUT_WIDTH, INPUT_HEIGHT), [0, 0, 0], 1, crop=False)

    # Sets the input to the network.
    net.setInput(blob)

    # Run the forward pass to get output of the output layers.
    outputs = net.forward(net.getUnconnectedOutLayersNames())
    return outputs


def post_process(input_image, outputs):
    # Lists to hold respective values while unwrapping.
    class_ids = []
    confidences = []
    boxes = []
    # Rows.
    rows = outputs[0].shape[1]
    image_height, image_width = input_image.shape[:2]
    # Resizing factor.
    x_factor = image_width / INPUT_WIDTH
    y_factor = image_height / INPUT_HEIGHT
    # Iterate through detections.
    for r in range(rows):
        row = outputs[0][0][r]
        confidence = row[4]
        # Discard bad detections and continue.
        if confidence >= CONFIDENCE_THRESHOLD:
            classes_scores = row[5:]
            # Get the index of max class score.
            class_id = np.argmax(classes_scores)
            #  Continue if the class score is above threshold.
            if (classes_scores[class_id] > SCORE_THRESHOLD):
                confidences.append(confidence)
                class_ids.append(class_id)
                cx, cy, w, h = row[0], row[1], row[2], row[3]
                left = int((cx - w / 2) * x_factor)
                top = int((cy - h / 2) * y_factor)
                width = int(w * x_factor)
                height = int(h * y_factor)
                box = np.array([left, top, width, height])
                boxes.append(box)
    indices = cv2.dnn.NMSBoxes(boxes, np.array(confidences), CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    df=pd.DataFrame(index=[i for i in range(len(indices))],columns=['X','Y','Category'])
    for i in indices:
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]             
        # Draw bounding box.   
        centre=(left+width//2,top+height//2)          
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        cv2.circle(input_image,centre,4,(0,255,255),2)
        centreRW=pixel_to_cm(centre[0],centre[1])
        centreRobot=cam2rbt(centreRW[0],centreRW[1])
        df.at[i,'X']=centreRobot[0]
        df.at[i,'Y']=centreRobot[1]
        df.at[i,'Category']=class_ids[i]
        df.to_csv('Data.csv')
        #print(robotAngles)
        # try:
        #     roi=input_image[top:top+height,left:left+width]
        #     roi=cv2.resize(roi,(256,256),interpolation=cv2.INTER_AREA)
        # except:
        #     pass
        # if np.sum([roi])!=0:
        #     roi=roi.astype('float')/255.0
        #     roi=img_to_array(roi)
        #     roi=np.expand_dims(roi,axis=0)
            
        #     prediction=model.predict(roi)[0]
        #     label=labels[prediction.argmax()]
        # Class label.                      
        label = "{}:{:.2f},{:.2f}".format(classes[class_ids[i]], centreRobot[0],centreRobot[1])
        #mappings[centreRobot]=class_ids[i]        
        # Draw label.             
        draw_label(input_image, label, left, top)
    return input_image



if __name__=='__main__':
    cap=cv2.VideoCapture(2)
    modelWeights = "best.onnx"
    net = cv2.dnn.readNet(modelWeights)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
    while True:
       ret,frame=cap.read()
       #frame=cv2.resize(frame,(INPUT_HEIGHT,INPUT_WIDTH))
       detections = pre_process(frame, net)
       img = post_process(frame.copy(), detections)
       t, _ = net.getPerfProfile()
       label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
       print(label)
       cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
       while True:
        cv2.imshow('Output', img)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
       break
        #i=2
        #if cv2.waitKey(1) & 0xFF==ord('s'):
         #   cv2.imwrite(f'{i}.jpg',img)
         #   i+=1
       #if cv2.waitKey(1) & 0xFF==ord('q'):
        #break
      #cv2.waitKey(0)
    """
    Put efficiency information. The function getPerfProfile returns       the overall time for inference(t) 
    and the timings for each of the layers(in layersTimes).
    """
    # t, _ = net.getPerfProfile()
    # label = 'Inference time: %.2f ms' % (t * 1000.0 /  cv2.getTickFrequency())
    # print(label)
    # cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE,  (0, 0, 255), THICKNESS, cv2.LINE_AA)
    # cv2.imshow('Output', img)
    # cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
