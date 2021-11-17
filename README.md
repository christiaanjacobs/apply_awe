# Applying multilingual AWE models

## Overview

Applying AWE models to extracted speech features. The experiemnts for producing the AWE models are described in:
- C. Jacobs, H. Kamper, and Y. Matusevych, "Acoustic word embeddings for zero-resource languages using self-supervised contrastive learning and multilingual adaptation," in *Proc. SLT*, 2021. [[arXiv](https://arxiv.org/abs/2103.10731)]

- C. Jacobs and H. Kamper, "Multilingual transfer of acoustic word embeddings improves when training on languages related to the target zero-resource language," in *Proc. Interspeech*, 2021. [[arXiv](https://arxiv.org/abs/2106.12834)]

## Data
The ```data``` directory contains speech features extracted from word segments, for example, MFCCs. The features must be stored in <em>filename.npz</em> file format. The file must store a list of sequences with size ```L x &#8902; x D``` where L is the number of word segments, &#8902; is the arbitrary length of a word segment and D is the dimensionality of a single speech feature.

Currently the data directory contains MFCCs from word segments in the English validation set from the NCHLT dataset. The code used to extract the features are given [here](https://github.com/christiaanjacobs/nchlt_awe/tree/master/features). The same feature extraction is applied [here](https://github.com/christiaanjacobs/globalphone_awe_pytorch) (currently this repo is better documented).

## Multilingual AWE models
The ```models``` directory contains multilingual AWE models. The directory indicates the languages the model is trained on.


Currently the directory contains two models trained using the ContrastiveRNN. 

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

