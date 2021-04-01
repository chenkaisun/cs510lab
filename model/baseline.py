from collections import deque, defaultdict, Counter
import json
import random
import time
import os
import numpy as np
import argparse
import gc
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data


class baseline(nn.Module):
    def __init__(self, args):
        super().__init__()

    def forward(self, input):
        texts = input.texts
        hid_texts = self.plm(**texts,
                             return_dict=True).pooler_output
