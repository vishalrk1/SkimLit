import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import tensorflow as tf
import numpy as np

def pad_sequences(sequences, max_seq_len=0):
    """Pad sequences to max length in sequence."""
    max_seq_len = max(max_seq_len, max(len(sequence) for sequence in sequences))
    padded_sequences = np.zeros((len(sequences), max_seq_len))
    for i, sequence in enumerate(sequences):
        padded_sequences[i][:len(sequence)] = sequence
    return padded_sequences

class SkimlitDataset(Dataset):
    def __init__(self, text_seq, line_num, total_line):
        self.text_seq = text_seq
        self.line_num_one_hot = line_num
        self.total_line_one_hot = total_line

    def __len__(self):
        return len(self.text_seq)

    def __str__(self):
        return f"<Dataset(N={len(self)})>"

    def __getitem__(self, index):
        X = self.text_seq[index]
        line_num = self.line_num_one_hot[index]
        total_line = self.total_line_one_hot[index]
        return [X, len(X), line_num, total_line]
  
    def collate_fn(self, batch):
        """Processing on a batch"""
        # Getting Input
        batch = np.array(batch)
        text_seq = batch[:,0]
        seq_lens = batch[:, 1]
        line_nums = batch[:, 2]
        total_lines = batch[:, 3]

        # padding inputs
        pad_text_seq = pad_sequences(sequences=text_seq) # max_seq_len=max_length

        # converting line nums into one-hot encoding
        line_nums = tf.one_hot(line_nums, depth=20)

        # converting total lines into one-hot encoding
        total_lines = tf.one_hot(total_lines, depth=24)

        # converting inputs to tensors
        pad_text_seq = torch.LongTensor(pad_text_seq.astype(np.int32))
        seq_lens = torch.LongTensor(seq_lens.astype(np.int32))
        line_nums = torch.tensor(line_nums.numpy())
        total_lines = torch.tensor(total_lines.numpy())
    
        return pad_text_seq, seq_lens, line_nums, total_lines

    def create_dataloader(self, batch_size, shuffle=False, drop_last=False):
        dataloader = DataLoader(dataset=self, batch_size=batch_size, collate_fn=self.collate_fn, shuffle=shuffle, drop_last=drop_last, pin_memory=True)
        return dataloader
