FROM nvidia/cuda:10.0-cudnn7-runtime-ubuntu18.04

RUN echo 1

RUN apt-get update

RUN apt-get install --fix-missing -y clang libpython-dev libblocksruntime-dev

RUN apt-get install -y curl

ARG SWIFT_TF_URL="https://storage.googleapis.com/s4tf-kokoro-artifact-testing/versions/v0.2/rc3/swift-tensorflow-RELEASE-0.2-cuda10.0-cudnn7-ubuntu18.04.tar.gz"

WORKDIR /swift

RUN curl -fSsL $SWIFT_TF_URL -o swift-tensorflow.tar.gz \
    && tar xzf swift-tensorflow.tar.gz --directory /swift

ENV PATH="/swift/usr/bin:$PATH"

WORKDIR /

RUN apt-get install -y wget

# Get Miniconda and make it the main Python interpreter
RUN apt-get install -y python3.6 libpython3.6 libpython3.6-dev
RUN apt-get install -y libxml2
RUN apt-get install -y libbsd-dev

ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

COPY init.swift init.swift

RUN swift init.swift

RUN apt-get install -y python3-pip

RUN pip3 install numpy
RUN pip3 install pandas
RUN pip3 install tensorflow-gpu==1.13.1
RUN pip3 install tensorflow-datasets