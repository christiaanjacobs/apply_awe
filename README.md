# Applying multilingual AWE models

## Overview
The provided code is for applying a multilingual AWE model to a set of speech features, specifically MFCCs, extracted from isolated word segments. The experiments for producing the AWE models are described in:
- C. Jacobs, H. Kamper, and Y. Matusevych, "Acoustic word embeddings for zero-resource languages using self-supervised contrastive learning and multilingual adaptation," in *Proc. SLT*, 2021. [[arXiv](https://arxiv.org/abs/2103.10731)]

- C. Jacobs and H. Kamper, "Multilingual transfer of acoustic word embeddings improves when training on languages related to the target zero-resource language," in *Proc. Interspeech*, 2021. [[arXiv](https://arxiv.org/abs/2106.12834)]

## Data
The ```data``` directory contains speech features extracted from word segments, for example, MFCCs. The features are stored in a numpy archive ```filename.npz```. The file stores a list of sequences with size ```L x * x D``` where ```L``` is the number of word segments, ```*``` is the arbitrary length of a word segment and ```D``` is the dimensionality of a single speech feature.

Currently, the data directory contains MFCCs from word segments in the English validation set from the NCHLT dataset. The code used to extract the features are given [here](https://github.com/christiaanjacobs/nchlt_awe/tree/master/features). The same feature extraction is applied [here](https://github.com/christiaanjacobs/globalphone_awe_pytorch) (currently this repo is better documented).

## Multilingual AWE models
The ```models``` directory contains multilingual AWE models. The directory indicates the languages the model is trained on. For example, the model with directory name ```models/contrastive/xho+nso+ssw+tsn+nbl+afr.gt/...``` is trained with an equal number of training examples from each language separated by ```+```. The corresponding language for each ISO language code (given here) are given in Table 1 in:
- E. Barnard, M. Davel, C. van Heerden, F. Wet, and J. Badenhorst, “The NCHLT speech corpus of the South African languages,” in *Proc. SLTU*, 2014. [[ResearchGate](https://www.researchgate.net/publication/301858320_The_nchlt_speech_corpus_of_the_south_african_languages)]


Currently, the directory contains two models trained using the ContrastiveRNN. 

## Install dependencies

You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch 1.4](https://pytorch.org/)

You can install all the other dependencies in a conda environment by running:

    conda env create -f env.yml
    conda activate awe

## Apply AWE model

Apply a multilingual ContrastiveRNN model to English validation data:

    ./apply_model_to_npz.py models/contrastive/xho+nso+ssw+tsn+nbl+eng.gt/18cdc6b389/final_model.pt \
        data/eng.dev.gt_words.npz \
        --output_npz_fn "outputs/eng_val_embed.npz"
  
 The output is a numpy archive storing an array of AWEs of size ```L x M```, where ```L``` is the number of embeddings and ```M``` is the dimensionality of the AWEs.
   
## Visualise AWEs

The ```visualise_embeddings.ipynb``` notebook can be used to visualise the embeddings. 
For example, the figure below shows the t-SNE visualisation of a selection of AWEs obtained by applying the multilingual ContrastiveRNN model to the English validation data. The figure shows that the AWEs of words from the same word type are more similar than the embeddings from different word types.
![tsne plot](https://github.com/christiaanjacobs/apply_awe/blob/master/outputs/tsne.png?raw=true)

