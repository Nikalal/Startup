#Deriving the latest base image
FROM python:3.8.8


#Labels as key value pair
LABEL Maintainer="Nick"


# Any working directory can be chosen as per choice like '/' or '/home' etc
# i have chosen /usr/app/src
WORKDIR /usr/src/app

#to COPY the remote file at working directory in container
COPY requirements.txt Sample_Consumer.py ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN apt-get update
RUN apt-get install vim -y
#RUN apt-get update

# Minimize image size 
#RUN (apt-get autoremove -y; apt-get autoclean -y)
#ENV QT_X11_NO_MITSHM=1
CMD ["bash"]

#CMD [ "python", "./consumer.py"]