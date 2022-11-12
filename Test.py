import cv2
import numpy as np
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

def draw_label(im, label, x, y):
    """Draw text onto image at location."""
    # Get text size.
    text_size = cv2.getTextSize(label, FONT_FACE, FONT_SCALE, THICKNESS)
    dim, baseline = text_size[0], text_size[1]
    # Use text size to create a BLACK rectangle.
    cv2.rectangle(im, (x,y), (x + dim[0], y + dim[1] + baseline), (0,0,0), cv2.FILLED);
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
    indices = cv2.dnn.NMSBoxes(boxes,np.array(confidences),NMS_THRESHOLD,CONFIDENCE_THRESHOLD)
    for i in indices:
        box = boxes[int(i)]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]             
        # Draw bounding box.   
        centre=(left+width//2,top+height//2)          
        cv2.rectangle(input_image, (left, top), (left + width, top + height), BLUE, 3*THICKNESS)
        cv2.circle(input_image,centre,4,(0,255,255),2)
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
        label = "{}:{:.2f}".format(classes[class_ids[int(i)]], confidences[int(i)])        
        # Draw label.             
        draw_label(input_image, label, left, top)
    return input_image

if __name__ == '__main__':
      # Load class names.
      classesFile = "coco.names"
      classes = ['Ripe','Green','Half-Ripe']
      # with open(classesFile, 'rt') as f:
      #       classes = f.read().rstrip('\n').split('\n')
      # # Load image.
      #frame = cv2.imread(‘traffic.jpg)
      # Give the weight files to the model and load the network using       them.
      modelWeights = "best.onnx"
      net = cv2.dnn.readNet(modelWeights)
      net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
      net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
      # Process image.
      cap=cv2.VideoCapture(0)
      while True:
          ret,frame=cap.read()
          #frame=cv2.resize(frame,(INPUT_HEIGHT,INPUT_WIDTH))
          detections = pre_process(frame, net)
          img = post_process(frame.copy(), detections)
          t, _ = net.getPerfProfile()
          label = 'Inference time: %.2f ms' % (t * 1000.0 / cv2.getTickFrequency())
          print(label)
          cv2.putText(img, label, (20, 40), FONT_FACE, FONT_SCALE, (0, 0, 255), THICKNESS, cv2.LINE_AA)
          cv2.imshow('Output', img)
          if cv2.waitKey(1) & 0xFF==ord('q'):
              break
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
