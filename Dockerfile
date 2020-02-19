FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

ARG DEBIAN_FRONTEND=noninteractive
LABEL maintainer="rappdw@gmail.com"

# install python 3.8.1
RUN apt-get update; \
    apt-get install -y \
        gcc \
        make \
        openssl \
        wget \
        zlib1g-dev \
        libreadline-gplv2-dev \
        libncursesw5-dev \
        libssl-dev \
        libsqlite3-dev \
        tk-dev \
        libgdbm-dev \
        libc6-dev \
        libbz2-dev \
		tcl \
		tk \
		libffi-dev \
		libgomp1 \
		libssl-dev \
    ; \
    apt-get clean; \
    rm -rf /var/tmp/* /tmp/* /var/lib/apt/lists/*; \
    cd \tmp; \
    wget https://www.python.org/ftp/python/3.8.1/Python-3.8.1.tgz;\
    tar -xvf Python-3.8.1.tgz; \
    cd Python-3.8.1; \
    ./configure; \
    make; \
    make install; \
    pip3 install -U pip; \
    ldconfig; \
    cd /usr/local/bin; \
	ln -Fs idle3 idle; \
	ln -Fs pydoc3 pydoc; \
	ln -Fs python3 python; \
	ln -Fs python3-config python-config

ENV PYTHON_VERSION=cp38 \
    CUDA_VERSION=cuda102 \
    PLATFORM=linux_x86_64 \
    BASE_URL='https://storage.googleapis.com/jax-releases'

RUN set -ex; \
    pip install -U $BASE_URL/$CUDA_VERSION/jaxlib-0.1.39-$PYTHON_VERSION-none-$PLATFORM.whl; \
    pip install -U jax

