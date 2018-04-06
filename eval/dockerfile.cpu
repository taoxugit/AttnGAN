FROM python:2

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install http://download.pytorch.org/whl/cpu/torch-0.3.1-cp27-cp27mu-linux_x86_64.whl 
RUN pip install torchvision 

COPY . /usr/src/app

ENV GPU False
ENV EXPORT_MODEL True

EXPOSE 8080

CMD ["python", "main.py"]


