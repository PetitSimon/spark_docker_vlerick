FROM python:3.9.12
LABEL Simon <simpetit1@gmail.com>:
ADD code.py . 
USER root
# Install apt packages and clean up cached files
RUN apt-get update && \
    apt-get install -y openjdk-8-jdk python3-venv && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "..." \

RUN python3 version RUN pip3 --version 

COPY dockerfile_imdbnotebook/ .

COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

CMD [ "python3", "spark_docker_vlerick/dockerfile_imdbnotebook/code.py"]


# Install the AWS CLI and clean up tmp files
#RUN wget https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip -O ./awscliv2.zip && \
   # unzip awscliv2.zip && \
   # ./aws/install
   # krm -rf ./aws awscliv2.zip


# For vscode
EXPOSE 3000
# for spark
EXPOSE 4040
