from kafka import KafkaConsumer
from people_detect import Detector
from kafka import TopicPartition
import numpy as np
from PIL import Image
import io
import cv2
import time
#consumer = KafkaConsumer(bootstrap_servers='localhost:9092',auto_offset_reset='earliest',
#                        group_id="consumer-group-zz")
consumer = KafkaConsumer(
        bootstrap_servers='broker_docker_new:9092',
        auto_offset_reset='latest',
        group_id="consumer-group-gg3")
consumer.assign([TopicPartition('new_topic', 0)])

if __name__ == "__main__":
    classesFilePath = './coco_names.txt'
    modelURL = 'http://download.tensorflow.org/models/object_detection/tf2/20200711/efficientdet_d0_coco17_tpu-32.tar.gz'
    detector = Detector(classesFilePath, modelURL)
    print("starting the consumer")
    for msg in consumer:
        bts = msg.value
        image = np.array(Image.open(io.BytesIO(bts))) 
        image = detector.predictImage(image)
        print(image)
        #cv2.imshow('image',image)
        #time.sleep(2)
        #cv2.waitKey(1)