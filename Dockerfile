FROM tensorflow/tensorflow:1.15.0-py3 as BASE
WORKDIR /usr/src/app
ADD . .
RUN pip install --upgrade pip
RUN pip install torch==0.4.1 
RUN pip install visdom 
RUN pip install Pillow==2.2.2
RUN pip install tqdm
RUN pip install torchvision==0.2.0
RUN apt-get update
RUN apt-get -yqq install vim 
ENTRYPOINT ["make", "start-visdom-server"] 
