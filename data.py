from utils import dump_file, load_file
import os
import torch
from torch.utils.data import Dataset
import json
from IPython import embed


# todo: load dataset
class OurDataset(Dataset):

    def __init__(self, args=None, filename="data/allMeSH_2021.json", tokenizer=None):
        if(args==None):
            self.instance = []
        else:
            args.cache_filename = os.path.splitext(filename)[0] + ".pkl"

            # Can cache data
            if args.use_cache and os.path.exists(args.cache_filename):
                print("Loading Cached Data...", args.cache_filename)
                data = load_file(args.cache_filename)
                self.instances = data['instances']
                for i, item in enumerate(self.instances):
                    self.instances[i]["tokenizer"] = tokenizer
                self.labelname2id = data['labelname2id']

            else:

                # load instance
                self.instances = []
                self.labelname2id = {}

                with open(filename, encoding="ISO-8859-1") as f:
                    for j, line in enumerate(f):
                        if args.debug and j > 30: break

                        line = line.strip()
                        if "meshMajor" in line:

                            # comma
                            ins = json.loads(line[:-1])

                            # ignore empty label
                            if len(ins["meshMajor"]):
                                # remove irelevent item
                                for i, lab in enumerate(ins["meshMajor"]):
                                    if lab not in self.labelname2id:
                                        self.labelname2id[lab] = len(self.labelname2id)
                                    ins["meshMajor"][i] = self.labelname2id[lab]

                                ins["id"] = j
                                ins["text"] = ins.pop("abstractText")
                                ins["label"] = ins.pop("meshMajor")

                                ins["tokenizer"] = tokenizer
                                self.instances.append(ins)

                                # print(line)
                                # print(self.instances[-1])
                                # embed()

                    for i, ins in enumerate(self.instances):
                        tmp = [0] * len(self.labelname2id)
                        for label_id in ins["label"]:
                            tmp[label_id] = 1
                        self.instances[i]["label"] = tmp

            args.out_dim = len(self.labelname2id)
            if args.cache_filename:
                dump_file({"instances": self.instances, "labelname2id": self.labelname2id}, args.cache_filename)

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


# todo: collate batch and send to gpu (like below)
class CustomBatch:
    def __init__(self, batch):
        # Collating
        self.ids = [f["id"] for f in batch]
        self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)

        tokenizer = batch[0]["tokenizer"]
        self.token_ids = tokenizer([f["text"] for f in batch], return_tensors='pt', padding=True, truncation=True,
                                   max_length=512)

        self.in_train = True
        # embed()

    def to(self, device):
        self.labels = self.labels.to(device)
        self.token_ids = {key: self.token_ids[key].to(device) for key in self.token_ids}

        return self

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.inp = self.inp.pin_memory()
        self.tgt = self.tgt.pin_memory()
        return self


def collate_wrapper(batch):
    return CustomBatch(batch)
