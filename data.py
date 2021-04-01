from utils import dump_file, load_file
import os
import torch
from torch.utils.data import Dataset

class OurDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, args, filename, tokenizer=None, modal_retriever=None):
        args.cache_filename = os.path.splitext(filename)[0] + ".pkl"

        if args.use_cache and os.path.exists(args.cache_filename):
            print("Loading Cached Data...", args.cache_filename)
            data = load_file(args.cache_filename)
            self.instances = data['instances']

        else:
            # load instance
            self.instances = []


        if args.cache_filename:
            dump_file({"instances": self.instances,}, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]

class CustomBatch:
    def __init__(self, batch):
        # Collating


        max_len = max([len(f["text"]) for f in batch])
        input_ids = [f["text"] + [0] * (max_len - len(f["text"])) for f in batch]
        input_mask = [[1.0] * len(f["text"]) + [0.0] * (max_len - len(f["text"])) for f in batch]
        self.texts = torch.tensor(input_ids, dtype=torch.long)
        self.texts_attn_mask = torch.tensor(input_mask, dtype=torch.float)

        # print("self.texts ",self.texts )
        # print("self.texts_attn_mask ",self.texts_attn_mask )

        # self.texts = batch[0]["tokenizer"]([f["text"] for f in batch], return_tensors='pt', padding=True)
        self.ids = [f["id"] for f in batch]
        self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)

        tokenizer = batch[0]["tokenizer"]

        g_data = Batch.from_data_list([f["ent"][0]['g'] for f in batch])
        g_data.x = torch.as_tensor(g_data.x, dtype=torch.long)
        self.ent1_g = g_data
        self.ent1_g_mask = torch.tensor([f["ent"][0]['g_mask'] for f in batch]).unsqueeze(-1)

        self.ent1_d = tokenizer([f["ent"][0]['t'] for f in batch], return_tensors='pt', padding=True)
        self.ent1_d_mask = torch.tensor([f["ent"][0]['t_mask'] for f in batch]).unsqueeze(-1)
        # print("self.ent1_d_mask ",self.ent1_d_mask )

        self.ent2_d = tokenizer([f["ent"][1]['t'] for f in batch], return_tensors='pt', padding=True)
        self.ent2_d_mask = torch.tensor([f["ent"][1]['t_mask'] for f in batch]).unsqueeze(-1)
        self.concepts = tokenizer(["chemical compound", "gene/protein"], return_tensors='pt', padding=True)

        self.ent1_pos = torch.tensor([f["ent"][0]['pos'] for f in batch], dtype=torch.long)
        self.ent2_pos = torch.tensor([f["ent"][1]['pos'] for f in batch], dtype=torch.long)
        self.in_train = True
        # embed()

    def to(self, device):
        self.texts = self.texts.to(device)
        self.texts_attn_mask = self.texts_attn_mask.to(device)
        self.labels = self.labels.to(device)

        self.ent1_g = self.ent1_g.to(device)
        self.ent1_g_mask = self.ent1_g_mask.to(device)
        self.ent1_d = {key: self.ent1_d[key].to(device) for key in self.ent1_d}
        self.ent1_d_mask = self.ent1_d_mask.to(device)

        self.ent2_d = {key: self.ent2_d[key].to(device) for key in self.ent2_d}
        self.ent2_d_mask = self.ent2_d_mask.to(device)

        self.concepts = {key: self.concepts[key].to(device) for key in self.concepts}

        self.ent1_pos = self.ent1_pos.to(device)
        self.ent2_pos = self.ent2_pos.to(device)

        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)
