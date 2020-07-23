# n2c2 Challenge 2019
This repository contains our contribution to the 2019 n2c2 challenge for track 1 ("n2c2/OHNLP Track on Clinical Semantic Textual Similarity"). The code in this repository can be used to reproduce the results in our paper "Extending BERT for Clinical Semantic Textual Similarity".

## How to Use the Code
- Make sure you have [Docker](https://www.docker.com/) installed on your system.
- Navigate to the [docker](docker) directory and build the image: `docker build --tag n2c2 .`
- Set the following environment variables:
    - `NLP_MODELS_PATH`: path to the directory where you downloaded the models. This directory should have the following structure (for the BERT model see [https://github.com/EmilyAlsentzer/clinicalBERT](https://github.com/EmilyAlsentzer/clinicalBERT) and the InferSent (and dependencies) models can be found at [https://github.com/facebookresearch/InferSent](https://github.com/facebookresearch/InferSent)):
    ```
    models
    └─ pretrained
       ├── document_embeddings
       └── word_embeddings
           ├── bert_models
           │   └── biobert_pretrain_output_all_notes_150000
           │       ├── bert_config.json
           │       ├── bert_model.ckpt.data-00000-of-00001
           │       ├── bert_model.ckpt.index
           │       ├── bert_model.ckpt.meta
           │       ├── graph.pbtxt
           │       ├── pytorch_model.bin
           │       └── vocab.txt
           ├── crawl-300d-2M
           │   ├── crawl-300d-2M.vec
           │   └── model_info.md
           ├── glove.840B.300d
           │   ├── glove.840B.300d.txt
           │   └── model_info.md
           ├── infersent1
           │   ├── infersent1.pkl
           │   └── model_info.md
           └── infersent2
               ├── infersent2.pkl
               └── model_info.md
    ```
     - `NLP_RAW_DATA`: path to the directory which ontains the raw challenge data with the following directory structure:
    ```
    challenge_data
    └── n2c2
        ├── clinicalSTS2019.test.gs.sim.txt
        ├── clinicalSTS2019.test.txt
        └── clinicalSTS2019.train.txt
    ```
    - `NLP_EXPERIMENT_PATH`: path to a directory which is used to store the results from the model.
- Run the container (also inside the [docker](docker) folder): `docker run --gpus all -it --rm -v ${PWD}/..:/workspace -v $NLP_MODELS_PATH:/mtc/models -v $NLP_RAW_DATA:/mtc/raw -v $NLP_EXPERIMENT_PATH:/mtc/experiment --name n2c2 n2c2`
- You can now execute the file [`generate_results.py`](mtc/generate_results.py) inside the container.
- If everything was successful, there should now be a subfolder named `submission_generation` which contains the resulting scores which we report in our paper.

## Copyright
Copyright © German Cancer Research Center (DKFZ), Division of Medical Image Computing (MIC).
Please make sure that your usage of this code is in compliance with the [code license](LICENSE).
