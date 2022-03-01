import cv2, os, time
import tensorflow as tf
import numpy as np
from tensorflow.python.keras.utils.data_utils import get_file
import imutils
import logging 

class Detector():

  def __init__(self,classesFilePath, modelURL):
      self.classesList, self.colorList = self.readClasses(classesFilePath)
      self.downloadModel(modelURL)
      self.model = self.loadModel()

  def readClasses(self,classesFilePath):
      with open(classesFilePath,'r') as f:
          classesList = f.read().splitlines()

      colorList = np.random.uniform(low=0, high=255, size = (len(classesList),3))

      return classesList, colorList    


  def downloadModel(self, modelURL):
      fileName = os.path.basename(modelURL)
      self.modelName = fileName[:fileName.index('.')]
      self.cacheDir = './pretrained_models'
      os.makedirs(self.cacheDir, exist_ok=True)

      get_file(fname=fileName,origin=modelURL, cache_dir=self.cacheDir, extract=True)       


  def loadModel(self):
      logging.info("Loading Model " + self.modelName +'\n')
      tf.keras.backend.clear_session()
      model = tf.saved_model.load('pretrained_models/datasets/efficientdet_d0_coco17_tpu-32/saved_model')
      print("Model " + self.modelName + " loaded successfully...")   
      return model 


  def createBoundingBox(self,image):
      inputTensor = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2RGB)
      inputTensor = tf.convert_to_tensor(inputTensor, dtype=tf.uint8)
      inputTensor = inputTensor[tf.newaxis,...]
      detections = self.model(inputTensor)
      bboxs = detections['detection_boxes'][0].numpy()
      classIndexes = detections['detection_classes'][0].numpy().astype(np.int32)
      classScores = detections['detection_scores'][0].numpy()

      imH, imW, imC = image.shape
      bboxIdx = tf.image.non_max_suppression(bboxs, classScores, max_output_size=50,iou_threshold=0.3, score_threshold=0.50)
      
      if( len(bboxs) !=0):
          for i in range(0,len(bboxIdx)):
              bbox = tuple(bboxs[i].tolist())
              classConfidence = round(100*classScores[i])
              classIndex = classIndexes[i]

              classLabelText = self.classesList[classIndex]
              classColor = self.colorList[classIndex]

              displayText = '{}: {}%'.format(classLabelText, classConfidence)

              ymin, xmin, ymax, xmax = bbox
              xmin, xmax, ymin, ymax = (xmin*imW, xmax*imW, ymin*imH, ymax*imH)
              xmin, xmax, ymin, ymax = int(xmin), int(xmax), int(ymin), int(ymax)
              #print(xmin,xmax,ymin,ymax)
              coordinates=[xmin,xmax,ymin,ymax]
              cv2.rectangle(image, (xmin,ymin), (xmax,ymax), color=classColor, thickness=1)

      return coordinates     

  def predictImage(self, imagePath):
    #image = cv2.imread(imagePath)
    #image = cv2.resize(image, (512, 512),interpolation = cv2.INTER_NEAREST)
    image = imutils.resize(imagePath, width=512)
    #bboxImage = self.createBoundingBox(image)
    bboxImage = self.createBoundingBox(image)
    print('done')
    #cv2.imshow(bboxImage) 
    return bboxImage
    


#classesFilePath = './coco_names.txt'
#modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
#detector = Detector(classesFilePath, modelURL)
#detector.predictImage('./frame0.jpg')