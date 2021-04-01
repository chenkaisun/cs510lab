import os
import pickle as pkl
import json
import numpy as np
import requests
from transformers import AutoTokenizer

def request_get(url, headers):
    try:
        r = requests.get(url, headers=headers)
        if r.ok:
            # print(r)
            return r
        else:

            print(r)
            return None
    except Exception as e:
        print(e)
        return None



def is_symmetric(g):
    return np.sum(np.abs(g.T - g)) == 0


def join(str1, str2):
    return os.path.join(str1, str2)


def get_ext(filename):
    return os.path.splitext(filename)[1]


def dump_file(obj, filename):
    if get_ext(filename) == ".json":
        with open(filename, "w+") as w:
            json.dump(obj, w)
    elif get_ext(filename) == ".pkl":
        with open(filename, "wb+") as w:
            pkl.dump(obj, w)
    else:
        print("not pkl or json")
        with open(filename, "w+", encoding="utf-8") as w:
            w.write(obj)


def load_file(filename):
    if get_ext(filename) == ".json":
        with open(filename, "r", encoding="utf-8") as r:
            res = json.load(r)
    elif get_ext(filename) == ".pkl":
        with open(filename, "rb") as r:
            res = pkl.load(r)
    return res


def mkdir(dir):
    if not os.path.isdir(dir):
        os.mkdir(dir)

def get_tokenizer(plm, save_dir="tokenizer/"):
    mkdir(save_dir)
    tk_name = plm.split("/")[-1].replace("-", "_") + "_tokenizer.pkl"
    tk_name=os.path.join(save_dir, tk_name)
    if not os.path.exists(tk_name):
        tokenizer = AutoTokenizer.from_pretrained(plm)
        dump_file(tokenizer, tk_name)
    return load_file(tk_name)