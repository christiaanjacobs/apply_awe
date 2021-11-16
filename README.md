# Applying multilingual AWE models to NCHLT data

## Overview

PyTorch implementation of multilingual acoustic word embedding approaches. The experiments are described in:
- C. Jacobs, H. Kamper, and Y. Matusevych, "Acoustic word embeddings for zero-resource languages using self-supervised contrastive learning and multilingual adaptation," in *Proc. SLT*, 2021. [[arXiv](https://arxiv.org/abs/2103.10731)]

The same Contrastive RNN implementation is used in:
- C. Jacobs and H. Kamper, "Multilingual transfer of acoustic word embeddings improves when training on languages related to the target zero-resource language," in *Proc. Interspeech*, 2021. [[arXiv](https://arxiv.org/abs/2106.12834)]

## Datasets

## Multilingual AWE models


## Install dependencies

You will require the following:

- [Python 3](https://www.python.org/downloads/)
- [PyTorch 1.4](https://pytorch.org/)
- [LibROSA](http://librosa.github.io/librosa/)
- [Cython](https://cython.org/)
- [tqdm](https://tqdm.github.io/)
- [speech_dtw](https://github.com/kamperh/speech_dtw/)
- [shorten](http://etree.org/shnutils/shorten/dist/src/shorten-3.6.1.tar.gz)

To install `speech_dtw` (required for same-different evaluation) and `shorten`
(required for processing audio), run `./install_local.sh`.

You can install all the other dependencies in a conda environment by running:

    conda env create -f environment.yml
    conda activate pyt1.4
