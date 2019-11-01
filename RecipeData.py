import torch
import torch.nn as nn
import torch.utils.data as data
import torch.utils.data.dataloader as dataloader
from torch.nn.utils.rnn import pad_sequence

# https://github.com/yunjey/seq2seq-dataloader/blob/master/data_loader.py
def collate_fn(data):
    """Creates mini-batch tensors from the list of tuples (src_seq, trg_seq).
    We should build a custom collate_fn rather than using default collate_fn,
    because merging sequences (including padding) is not supported in default.
    Seqeuences are padded to the maximum length of mini-batch sequences (dynamic padding).
    Args:
        data: list of tuple (src_seq, trg_seq).
            - src_seq: torch tensor of shape (?); variable length.
            - trg_seq: torch tensor of shape (?); variable length.
    Returns:
        src_seqs: torch tensor of shape (batch_size, padded_length).
        src_lengths: list of length (batch_size); valid length for each padded source sequence.
        trg_seqs: torch tensor of shape (batch_size, padded_length).
        trg_lengths: list of length (batch_size); valid length for each padded target sequence.
    """
    def merge(sequences):
        lengths = [len(seq) for seq in sequences]
        padded_seqs = torch.zeros(len(sequences), max(lengths)).long()
        for i, seq in enumerate(sequences):
            end = lengths[i]
            padded_seqs[i, :end] = seq[:end]
        return padded_seqs, lengths

    # sort a list by sequence length (descending order) to use pack_padded_sequence
#     print(data[0]) # list of tuples
    data.sort(key=lambda x: len(x[0]), reverse=True)

    # seperate source and target sequences
    src_seqs, trg_seqs = zip(*data)

    # merge sequences (from tuple of 1D tensor to 2D tensor)
    src_seqs, src_lengths = merge(src_seqs)
    # target sequence for us is a single tensor so we do not need to 
    # merge it
    #trg_seqs, trg_lengths = merge(trg_seqs)
    trg_seqs = torch.as_tensor(trg_seqs)
    return src_seqs, src_lengths, trg_seqs #, trg_lengths


class RecipeData(data.Dataset):
    
    def __init__(self, df):
        super(RecipeData, self).__init__()
        self.df = df
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        X = torch.as_tensor(self.df.Ingredient_Numeric.iloc[idx])
        y = torch.as_tensor(self.df.Recipe_id_numeric.iloc[idx])
        return X,y
    