#FROM pytorch/pytorch:1.0-cuda10.0-cudnn7-runtime
FROM pytorch/pytorch:1.9.0-cuda10.2-cudnn7-runtime

RUN chmod 777 /var
WORKDIR /var
ADD mnist_cifar_ddp_gpu.py /var
ADD requirements.txt /var
RUN pip install --no-cache-dir -r /var/requirements.txt

ENTRYPOINT ["python3", "/var/mnist_cifar_ddp_gpu.py"]
