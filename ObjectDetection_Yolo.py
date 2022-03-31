import cv2
import numpy as np

whT = 320
confThreshold = 0.5
nmsThreshold = 0.3
cap = cv2.VideoCapture(0)

classFile = 'Resources/coco.names.txt'
classNames = []
with open(classFile, 'rt') as f:
    classNames = f.read().rstrip('\n').split('\n')  #to store each object names present in coco.names to classNames array

modelConfiguration = "Resources/yolov3.cfg.txt"
modelWeights = "Resources/yolov3.weights"

net = cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights) #creating the network
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV) #Declaring OpenCv is used as the backend
net.setPreferableBackend(cv2.dnn.DNN_TARGET_CPU)  #To use CPU for the network

def findObjects(outputs, img):
    hT, wT, cT = img.shape
    bbox = [] #to store the x,y , width and height of bounding box
    classIds = [] #To store all the class ids
    confs = [] #To store confidence values of each class

    for output in outputs:                  #to loop through each output layer
        for det in output:                  #to loop through each box of an output layer
            scores = det[5:]
            classId = np.argmax(scores)     # to get the id of the class with maximum confidence
            confidence = scores[classId]    # Stores the highest confidence value
            if confidence > confThreshold:
                w,h = int(det[2]*wT), int(det[3] *hT)           #Obtaining the width and height of the box in pixels
                x,y = int((det[0]*wT)-w/2), int((det[1]*hT)-h/2)  #Obtaining the x and y values of the box in pixels
                bbox.append([x,y,w,h])
                classIds.append(classId)
                confs.append(float(confidence))

    #Now we have the bounding box points of the objects detected
    #And we need to avoid the scenario where an inner box detects the image already detected by the outer box
    indices =  cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold) #Lower the value of nms, the more agressive and lesser the no of boxes
    for i in indices:
        box = bbox[i]
        print(box)
        x,y,w,h = box[0], box[1], box[2], box[3]
        cv2.rectangle(img,(x,y),(x+w, y+h),(255,0,255),2)
        cv2.putText(img,f'{classNames[classIds[i]].upper()} {int(confs[i]*100)}%', (x,y-10), cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,0,255), 2)

while True:
    success, img = cap.read()

    blob = cv2.dnn.blobFromImage(img, 1/255, (whT, whT),[0,0,0],1,crop=False)  #To convert the image to blob format
    net.setInput(blob)

    layerNames = net.getLayerNames()  # To retrieve the layer names
    # print(net.getUnconnectedOutLayers())  #Retrives the index of the three output layers of the network
    outputNames = [layerNames[i-1] for i in net.getUnconnectedOutLayers()] #Picks the output layer names based on indices returned

    outputs = net.forward(outputNames)  # Gives the output of the output layers
    # ex: for shape -(300,85) - 85 is Xcenter, Ycenter, width, height, confidence of object presence, the probabililty of presence of the 80 classes
    # and 300 is the number of boxes
    print(outputs[0].shape)
    print(outputs[1].shape)
    print(outputs[2].shape)
    findObjects(outputs, img)

    cv2.imshow("Screen", img)
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break
