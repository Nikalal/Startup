import sys
import time
import cv2
from kafka import KafkaProducer

topic = 'quickstart'

def publish_video():
    """
    Publish given video file to a specified Kafka topic. 
    Kafka Server is expected to be running on the localhost. Not partitioned.
    
    :param video_file: path to video file <string>
    """
    # Start up producer
    producer = KafkaProducer(bootstrap_servers='localhost:19092')

    # Open file
    video = cv2.VideoCapture('video.mp4')
    
    print('publishing video...')

    while(video.isOpened()):
        success, frame = video.read()

        # Ensure file was read successfully
        if not success:
            print("bad read!")
            break
        
        # Convert image to png
        ret, buffer = cv2.imencode('.jpg', frame)

        # Convert to bytes and send to kafka
        producer.send(topic, buffer.tobytes())

        time.sleep(0.2)
        
    video.release()
    print('publish complete')



if __name__ == '__main__':
    """
    Producer will publish to Kafka Server a video file given as a system arg. 
    Otherwise it will default by streaming webcam feed.
    """
    #if(len(sys.argv) > 1):
    #    video_path = sys.argv[1]
    #    publish_video(video_path)
    #else:
    print("publishing feed!")
    publish_video()