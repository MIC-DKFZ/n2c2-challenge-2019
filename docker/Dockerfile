FROM pytorch/pytorch:1.1.0-cuda10.0-cudnn7.5-devel

# Required folders for the application which must be set as shared volume during runtime
RUN mkdir --parents /mtc/models /mtc/raw /mtc/experiment
ENV NLP_MODELS_PATH=/mtc/models
ENV NLP_RAW_DATA=/mtc/raw
ENV NLP_EXPERIMENT_PATH=/mtc/experiment

# Python packages
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt \
 && git clone https://github.com/NVIDIA/apex \
 && cd apex \
 && git checkout de6378f5dae8fcf2879a4be8ecea8bbcb9e59d53 \
 && pip install -v --no-cache-dir --global-option="--cpp_ext" --global-option="--cuda_ext" ./

# Additional dependencies
RUN apt-get update && apt-get install -y openjdk-8-jre zip \
 && pip install gdown \
 && cd /mtc \
 && gdown https://drive.google.com/uc?id=1OMsz4n1rlOFbO2Ckc5kOoty0CddgTR_t \
 && unzip Medex_UIMA_1.3.8.zip \
 && rm Medex_UIMA_1.3.8.zip \
 && git clone https://github.com/facebookresearch/InferSent.git

ENV MEDEX_PATH=/mtc/Medex_UIMA_1.3.8
ENV PATH_TO_INFERSENT=/mtc/InferSent
ENV PYTHONPATH=/workspace
