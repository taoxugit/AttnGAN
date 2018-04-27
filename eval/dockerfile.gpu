FROM nvidia/cuda

RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install -y \
    python-pip \
    python2.7 \
    && apt-get autoremove \
    && apt-get clean

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install http://download.pytorch.org/whl/cu90/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
RUN pip install torchvision 

COPY . /usr/src/app

ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV GPU True
ENV EXPORT_MODEL False

EXPOSE 8080

CMD ["python", "main.py"]