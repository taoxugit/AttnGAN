FROM python:2

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY requirements.txt /usr/src/app/
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p /usr/src/app/code
COPY code /usr/src/app/code

RUN mkdir -p /usr/src/app/DAMSMencoders
COPY DAMSMencoders /usr/src/app/DAMSMencoders

RUN mkdir -p /usr/src/app/models
COPY models /usr/src/app/models

RUN mkdir -p /usr/src/app/data
COPY data /usr/src/app/data

EXPOSE 8080

#RUN python code/main.py --cfg code/cfg/eval_bird.yml

#RUN python main.py --cfg cfg/eval_bird.yml --gpu -1

CMD ["sh"]
