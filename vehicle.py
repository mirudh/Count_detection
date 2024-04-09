import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('scooters-5638.mp4')

def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()
area1=set()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    results=model(frame)
    frame=np.squeeze(results.render())
    list=[]
    l1=[]
    l2=[]
    l3=[]
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        b=str(row['name'])
        if ('car') or ('bicycle') or ('motorbike') in b:
            list.append([x1,y1,x2,y2])
            #area1.add([x1,y1,x2,y2])
            boxes_ids=tracker.update(list)
            for box_id in boxes_ids:
                x,y,w,h,id=box_id
                cv2.rectangle(frame,(x,y),(w,h),(0,0,255),2)
        
        if 'car' in b:
            a = b.count('car')
            l1.append(a)
        if 'bicycle' in b:
            c = b.count('bicycle')
            l2.append(c)
        if 'motorcycle' in b:
            d = b.count('motorbike')
            l3.append(a)
            
    cv2.imshow('FRAME',frame)
    frame=np.squeeze(results.render())
    cv2.waitKey(1)
    print('car found:',len(l1))
    print('bicycle found:',len(l2))
    print('motorcycle found:',len(l3))
    
cap.release()
cv2.destroyAllWindows()