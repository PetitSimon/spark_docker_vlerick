FROM python:3.9

WORKDIR /app

COPY . /app
 
RUN pip install -r requirements.txt

RUN python3 version RUN pip3 --version 


RUN apt-get update && \
    apt-get install -y default-jdk && \
    apt-get install -y ant && \
    apt-get clean;

EXPOSE 8080

ENV NAME World

CMD ["python", "/app/main.py"]


# For vscode
EXPOSE 3000
# for spark
EXPOSE 4040
