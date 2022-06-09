import torch
import random
import numpy as np


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def collate_fn(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    i_s = [f["i_s"] for f in batch]
    t_s = [f["t_s"] for f in batch]
    l_s = [f["l_s"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    i_s = torch.tensor(i_s, dtype=torch.long)
    t_s = torch.tensor(t_s, dtype=torch.long)
    l_s = torch.tensor(l_s, dtype=torch.long)
    sample_ids = [f["sample_ids"] for f in batch]
    #output = (input_ids, input_mask, labels, i_s, t_s, sample_ids)
    output = (input_ids, input_mask, labels, i_s, t_s, l_s, sample_ids)
    return output
