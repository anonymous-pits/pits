FROM nvcr.io/nvidia/pytorch:21.06-py3
RUN apt-get update && apt-get upgrade -y
RUN mkdir /root/pits 
COPY . /root/pits
RUN cd /root/pits && \
    python3 -m pip uninstall -y \
    tensorboard \
    tensorboard-plugin-dlprof \
    nvidia-tensorboard \
    nvidia-tensorboard-plugin-dlprof \
    jupyter-tensorboard \
    && \
    python3 -m pip --no-cache-dir install -r requirements.txt \
        && \
    apt update && \
    apt install -y \
        tmux \
        htop \
        ncdu && \
    apt clean && \
    apt autoremove && \
    rm -rf /var/lib/apt/lists/* /tmp/* && \
    cd /root/pits/monotonic_align && \
    python3 setup.py build_ext --inplace
WORKDIR /root/pits
EXPOSE 6006
