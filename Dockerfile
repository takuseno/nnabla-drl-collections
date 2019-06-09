FROM nnabla/nnabla-ext-cuda:py36-cuda92-v1.0.17

ENV LC_ALL C.UTF-8
ENV LANG C.UTF-8

COPY requirements.txt /tmp

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential software-properties-common cmake curl python3-dev \
    x11-apps zlib1g zlib1g-dev graphviz libgl1-mesa-dri libgl1-mesa-glx \
    libglu1-mesa-dev xvfb x11-utils libasio-dev git pkg-config python3-tk \
    libsm6 libxext6 libxrender1 git libpcre3-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/* && \
    pip --no-cache-dir install -r /tmp/requirements.txt && \
    pip --no-cache-dir install pybullet roboschool

WORKDIR /home/app

EXPOSE 6006

CMD tail -f /dev/null
