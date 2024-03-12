FROM python:3.9

RUN apt-get update && apt-get install -y libgl1-mesa-glx

WORKDIR /

COPY . .

RUN pip install -r requirements.txt

CMD [ "python", "-u", "/demo.py" ]
