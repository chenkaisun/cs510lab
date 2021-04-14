from utils import dump_file, load_file
import os
import torch
from torch.utils.data import Dataset
import json
from IPython import embed
import pickle

def file_len(fname):
    import subprocess
    p = subprocess.Popen(['wc', '-l', fname], stdout=subprocess.PIPE,
                                              stderr=subprocess.PIPE)
    result, err = p.communicate()
    if p.returncode != 0:
        raise IOError(err)
    return int(result.strip().split()[0])

# todo: load dataset
class OurDataset(Dataset):

    def __init__(self, args=None, filename="data/allMeSH_2021.json", tokenizer=None):
        if(args==None):
            self.instances = []
        else:
            args.cache_filename_prefix = os.path.splitext(filename)[0]

            # Can cache data
            if args.use_cache and os.path.exists(args.cache_filename_prefix + "_instances.pkl") and os.path.exists(args.cache_filename_prefix + "_labelname2id.pkl"):
                print("Loading Cached Data...", args.cache_filename_prefix)
                with open(args.cache_filename_prefix + "_instances.pkl", "rb") as fin:
                    self.instances = pickle.load(fin)
                with open(args.cache_filename_prefix + "_labelname2id.pkl", "rb") as fin:
                    self.labelname2id = pickle.load(fin)
                for i, item in enumerate(self.instances):
                    self.instances[i]["tokenizer"] = tokenizer
            else:

                # load instance
                self.instances = []
                self.labelname2id = {}

                total_line = file_len(filename)
                print("Start processing raw data. Total lines: {}".format(total_line))

                with open(filename, encoding="ISO-8859-1") as f:
                    for j, line in enumerate(f):

                        if j > 500000: break

                        if j % 100000 == 0:
                            print("[{:.2f}%] {}/{}".format(100.0*j/total_line, j, total_line), end="\r")

                        if args.debug and j > 10000: break

                        line = line.strip()
                        if "meshMajor" in line:

                            # comma
                            try:
                                # the last line contain redundant "]}"
                                ins = json.loads(line[:-1] if j != total_line else line[:-2])
                            except Exception as e:
                                print("Error in line {}/{}".format(j, total_line))
                                print(line[:-3])
                                raise e

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
                        ind = [[0 for label_id in ins["label"]], ins["label"]]
                        v = [1 for label_id in ins["label"]]
                        tmp = torch.sparse_coo_tensor(ind, v, (1, len(self.labelname2id)))
                        self.instances[i]["label"] = tmp
                
                print("Done processing raw data.")

                if args.cache_filename_prefix and False:    
                    with open(args.cache_filename_prefix + "_instances.pkl", "wb") as fout:
                        pickle.dump(self.instances, fout)
                    with open(args.cache_filename_prefix + "_labelname2id.pkl", "wb") as fout:
                        pickle.dump(self.labelname2id, fout)
                    print("Cached in {}".format(args.cache_filename_prefix))

            args.out_dim = len(self.labelname2id)
            print(len(self.labelname2id))
            
    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        return self.instances[idx]


# todo: collate batch and send to gpu (like below)
class CustomBatch:
    def __init__(self, batch):
        # Collating
        self.ids = [f["id"] for f in batch]
        # self.labels = torch.tensor([f["label"] for f in batch], dtype=torch.long)
        self.labels = torch.cat([f["label"].to_dense() for f in batch], dim=0).to(torch.long)

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
