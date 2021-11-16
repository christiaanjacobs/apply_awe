#!/home/christiaan/anaconda3/envs/myenv/bin/python
"""
Encode the given NumPy archive using the specified model.
"""

from datetime import datetime
from os import path
import argparse
import pickle
import numpy as np
import os
import sys
import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence

import data_io
import models

#-----------------------------------------------------------------------------#
#                              UTILITY FUNCTIONS                              #
#-----------------------------------------------------------------------------#

def check_argv():
    """Check the command line arguments."""
    parser = argparse.ArgumentParser(
        description=__doc__.strip().split("\n")[0], add_help=False
        )
    parser.add_argument("model_fn", type=str, help="model checkpoint filename")
    parser.add_argument("npz_fn", type=str, help="the NumPy archive to encode")
    parser.add_argument(
        "--output_npz_fn", type=str,
        help="if provided, the output is written to this NumPy archive "
        "instead of the model directory"
        )
    if len(sys.argv) == 1:
        parser.print_help()
        sys.exit(1)
    return parser.parse_args()


def collate_fn(batch):
    """
    batch = (data, lengths), data is shape (seq_len, feat_dim)
    """
    seqs = [x[0] for x in batch]
    batch_seqs_padded = pad_sequence(seqs, batch_first=True) # pad variable length sequences to max seq length
    lengths = [x[1] for x in batch]
    batch_lengths = torch.LongTensor(lengths)
    keys = [x[2] for x in batch]
    return batch_seqs_padded, batch_lengths, keys


#-----------------------------------------------------------------------------#
#                                   Dataset                                   #
#-----------------------------------------------------------------------------#
class npz_data(Dataset):
    def __init__(self, x_data, lengths, keys):
        self.x_data = x_data
        self.lengths = lengths
        self.keys = keys

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        return self.x_data[idx], self.lengths[idx], self.keys[idx]


#-----------------------------------------------------------------------------#
#                            APPLY MODEL FUNCTIONS                            #
#-----------------------------------------------------------------------------#

def apply_model(model_fn, npz_fn):
    # Load model options
    model_dir = path.split(model_fn)[0]
    options_dict_fn = path.join(model_dir, "options_dict.pkl")
    print("Reading:", options_dict_fn)
    with open(options_dict_fn, "rb") as f:
        options_dict = pickle.load(f)

    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  
    print('Device:', device)  
    options_dict['device'] = device

    # Load data
    x_data, labels, lengths, keys, speakers = data_io.load_data_from_npz(npz_fn)

    # Truncate and limit dimensionality
    data_io.trunc_and_limit_dim(x_data, lengths, options_dict["n_input"], options_dict["max_length"])

    # Load model
    model = models.encoder_rnn(options_dict, options_dict["batch_size"])
    model.load_state_dict(torch.load(model_fn)["state_dict"], strict=False)
    model.to(device)
    model.eval()

    # Encapuslate data in Dataset and apply Dataloader
    dataset = npz_data(x_data, lengths, keys)
    dataloader = DataLoader(dataset, batch_size=10000,
        shuffle=False, drop_last=False, collate_fn=collate_fn)

    # Iterate
    embed_dict = {}
    for batch_seqs_padded, batch_lengths, keys in dataloader:
        print(batch_seqs_padded.shape)
        batch_seqs_padded  = batch_seqs_padded.to(device)
        np_z = model(batch_seqs_padded, batch_lengths).cpu().detach().numpy()
        # break # Single batch

        for i, utt_key in enumerate(keys):
            embed_dict[utt_key] = np_z[i]

    return embed_dict
#-----------------------------------------------------------------------------#
#                                MAIN FUNCTION                                #
#-----------------------------------------------------------------------------#

def main():
    args = check_argv()
    
    # Embed data
    embed_dict = apply_model(args.model_fn, args.npz_fn)

    # Save embeddings
    model_dir, model_fn = path.split(args.model_fn)
    if args.output_npz_fn is None:
        npz_fn = path.join(
            model_dir, path.splitext(args.model_fn)[0] + "." +
            path.split(args.npz_fn)[-1]
            )
    else:
        npz_fn = args.output_npz_fn
    print("Writing:", npz_fn)
    np.savez_compressed(npz_fn, **embed_dict)
    print(datetime.now())


if __name__ == "__main__":
    main()
