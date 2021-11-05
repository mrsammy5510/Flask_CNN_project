from types import MethodDescriptorType
#from flask import Blueprint
import numpy as np
import os
import cv2
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import numpy as np

model_path = './model/mask_detector.h5'
prototxtpath = "./model/deploy.prototxt"
weightspath = "./model/res10_300x300_ssd_iter_140000.caffemodel"


#mask_detector_api = Blueprint("mask_detector_api", __name__)



def show_img(img):
  cv2.imshow('test',img)
  cv2.waitKey(0)
  cv2.destoryAllWindows()



#讓模型檢測照片是否有戴口罩
#需要的檔案只有
'''
deploy.prototxt
mask_detector.h5
res10_300x300_ssd_iter_140000.caffemodel
'''
def mask_detect_showimg(filename,img_path):
  image_path = img_path
  set_confidence = 0.5
  
  print("loading face detector model")
  net = cv2.dnn.readNet(prototxtpath,weightspath)

  print("loading face mask detector model")
  model = load_model(model_path)

  img = cv2.imread(image_path)
# origin_img = img.copy()
  (h,w) = img.shape[:2]

  blob = cv2.dnn.blobFromImage(img,1.0,(300,300),(104.0,177.0,123.0))
  print("computing face detections...")
  net.setInput(blob)
  detections = net.forward()

  for i in range(0,detections.shape[2]):
    confidence = detections[0,0,i,2]
    if confidence > set_confidence:
      box = detections[0,0,i,3:7] * np.array([w,h,w,h])
      (startX,startY,endX,endY) = box.astype("int")
      (startX,startY) = (max(0,startX),max(0,startY))
      (endX,endY) = (min(w-1,endX),min(h-1,endY))
      face = img[startY:endY,startX:endX]
      face = cv2.cvtColor(face,cv2.COLOR_BGR2RGB)
      face = cv2.resize(face,(224,224))
      face = img_to_array(face)
      face = preprocess_input(face)
      face = np.expand_dims(face,axis=0)
      (mask,withoutmask) = model.predict(face)[0]

      label = "Mask" if mask > withoutmask else "No mask"
      color = (0,255,0) if label == "Mask" else (0,0,255)
      label = "{}:{:.2f}%".format(label,max(mask,withoutmask)*100)
      cv2.putText(img, label, (startX,startY-10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
      cv2.rectangle(img, (startX,startY), (endX,endY), color, 2)
      
  cv2.imwrite("./static/predict_"+filename+".jpg", img)
  #show_img(img)


# if __name__ == '__main__':
#   mask_detect_showimg('./chen.jpg')