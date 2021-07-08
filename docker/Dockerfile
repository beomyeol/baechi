FROM continuumio/miniconda3

RUN apt update && apt install gcc -y

RUN conda install -y \
      python=3.6 \
      numpy=1.16 \
      bazel=0.20.0 \
      networkx \
      future \
      matplotlib \
      cvxopt \
      scikit-learn \
      cudatoolkit=10.0 \
      cudnn \
      cupti

COPY tensorflow-1.12.3-cp36-cp36m-linux_x86_64.whl .
RUN pip install ./tensorflow-1.12.3-cp36-cp36m-linux_x86_64.whl
RUN pip install -f https://download.mosek.com/stable/wheel/index.html Mosek==8.1.82

RUN mkdir /baechi
WORKDIR /baechi

COPY BUILD License.txt README.md WORKSPACE train.py ./
COPY placer placer
COPY image_classifier image_classifier
COPY nmt nmt
COPY third_party third_party
COPY utils utils

