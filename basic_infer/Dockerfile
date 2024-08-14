# pull the nvidia image
FROM nvcr.io/nvidia/pytorch:22.07-py3

# prohibit interaction during this build
ARG DEBIAN_FRONTEND=noninteractive

# replace apt source
RUN cp /etc/apt/sources.list /etc/apt/sources.list.bak
RUN sed -i 's/archive.ubuntu.com/mirrors.aliyun.com/g' /etc/apt/sources.list

# install python libs
RUN pip install torch==2.0.0 torchvision==0.15.1  -i https://pypi.tuna.tsinghua.edu.cn/simple
RUN pip install nebulae ptflops opencv-python-headless==4.5.5.64 -i https://pypi.tuna.tsinghua.edu.cn/simple

