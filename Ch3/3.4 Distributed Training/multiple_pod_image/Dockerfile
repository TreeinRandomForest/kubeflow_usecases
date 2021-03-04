FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime

RUN chmod 777 /var
WORKDIR /var
ADD ccfraud_distributed.py /var
ADD creditcard.csv /var
ADD requirements.txt /var
RUN pip install --no-cache-dir -r /var/requirements.txt
RUN mkdir /data
RUN chmod 777 /data

ENTRYPOINT ["python", "/var/ccfraud_distributed.py"]
