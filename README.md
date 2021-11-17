# Applying multilingual AWE models

## Overview

PyTorch implementation of multilingual acoustic word embedding approaches. The experiments are described in:
- C. Jacobs, H. Kamper, and Y. Matusevych, "Acoustic word embeddings for zero-resource languages using self-supervised contrastive learning and multilingual adaptation," in *Proc. SLT*, 2021. [[arXiv](https://arxiv.org/abs/2103.10731)]

The same Contrastive RNN implementation is used in:
- C. Jacobs and H. Kamper, "Multilingual transfer of acoustic word embeddings improves when training on languages related to the target zero-resource language," in *Proc. Interspeech*, 2021. [[arXiv](https://arxiv.org/abs/2106.12834)]

## Data
The <em>data</em> directory contains speech features extracted from word segments, for example MFCCs. The features must be in n the format "[filename].npz" stroing  list of sequences with size L x &#955; numpy array of (* is any number of trailing dimensions)
The <em>data</em> directory contains the data to apply the AWE models. Each .npz file contains the extracted MFCCs of isolaed word segments.
The features are extracted as in ... described in ...
Currently only contains the extracted features from he English development set. 

## Multilingual AWE models
The <em>models</em> directory contains multilingual AWE models. The directory indicates the languages the model is trained on.
Only contains models trained using the ContrastiveRNN in ... and ...

## Install dependencies

You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch 1.4](https://pytorch.org/)

You can install all the other dependencies in a conda environment by running:

    conda env create -f env.yml
    conda activate awe

## Apply AWE model

Apply a multilingual model to extracted speech features

    ./apply_model_to_npz.py models/contrastive/xho+nso+ssw+tsn+nbl+eng.gt/18cdc6b389/final_model.pt \
        data/eng.dev.gt_words.npz \
        --output_npz_fn "outputs/eng_val_embed.npz"
   
   
## Visualise AWEs

The <em>visualise_embeddings.ipynb</em> notebook can be used to visualise the embeddings. 
For example, the figure below shows the t-SNE visualisation of AWEs obtained by applying the multilingual ContrastiveRNN model to the English validation data.
The multilingual ContrastiveRNN AWE model is trained on: isiXhosa, Sepedi, siSwati, Setswana, isiNdebele, English.  

![tsne plot](https://github.com/christiaanjacobs/apply_awe/blob/master/outputs/tsne.png?raw=true)

