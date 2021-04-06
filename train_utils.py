import logging
import os
import random
import sys

import numpy as np
import torch
from torch import optim

from model.load_model import get_model, load_model_from_path
from utils import mkdir
from torch.utils.tensorboard import SummaryWriter
import numpy


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    numpy.random.seed(worker_seed)
    random.seed(worker_seed)


def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    mkdir(args.experiment_path)
    output_file_handler = logging.FileHandler(args.experiment_path + args.exp + ".txt")
    stdout_handler = logging.StreamHandler(sys.stdout)
    logger.addHandler(output_file_handler)
    logger.addHandler(stdout_handler)
    return logger


def setup_common(args):
    print("setup_common")
    mkdir("model")
    mkdir("model/states")

    args.device = gpu_setup(use_gpu=args.use_gpu)
    if "cpu" in str(args.device): args.use_amp = 0

    if "-tiny" in args.plm:
        args.plm_hidden_dim = 128
    elif "-mini" in args.plm:
        args.plm_hidden_dim = 256
    elif "-small" in args.plm:
        args.plm_hidden_dim = 512
    elif "-medium" in args.plm:
        args.plm_hidden_dim = 512
    else:
        args.plm_hidden_dim = 768

    model = get_model(args)
    # view_model_param(args, model)

    downstream_layers = ["extractor", "bilinear", "combiner", "gnn", "msg_encoder","query_encoder", "final"]
    optimizer = get_optimizer(args, model, downstream_layers)
    # print(model.named_parameters())
    # print("model", model)

    print("Optimizer built")
    model, optimizer, args.start_epoch, args.best_dev_score = load_model_from_path(model, optimizer, args)
    print("Model moved to gpu")

    args.logger = get_logger(args)
    args.writer = SummaryWriter(log_dir=args.experiment_path + args.exp + "/")
    args.logger.debug("=====begin of args=====")

    arg_dict = vars(args)
    args.important_hparams = {}
    for key in sorted(arg_dict.keys()):
        args.logger.debug(f"{key}: {arg_dict[key]}")
    args.logger.debug("=====end of args=====")

    return args, model, optimizer


def gpu_setup(use_gpu=True, gpu_id=0, use_random_available=True):
    print("Setting up GPU")
    if not use_random_available:
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

    num_gpus = 1
    if torch.cuda.is_available() and use_gpu:
        print(f"{torch.cuda.device_count()} GPU available")
        print('cuda available with GPU:', torch.cuda.get_device_name(0))

        device = torch.device("cuda")

    else:
        if not torch.cuda.is_available():
            print('cuda not available')
        device = torch.device("cpu")
    return device


def view_model_param(args, model):
    # model = get_model(args)
    total_param = 0
    print("MODEL DETAILS:\n")
    # print(model)
    for param in model.parameters():
        # print(param.data.size())
        total_param += np.prod(list(param.data.size()))
    print('MODEL/Total parameters:', args.model_name, total_param)
    return total_param


def get_optimizer(args, model, downstream_layers):
    optimizer_grouped_parameters = [
        {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in downstream_layers)], },
        {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in downstream_layers)],
         "lr": args.lr}
    ]
    return optim.AdamW(optimizer_grouped_parameters,
                       lr=args.plm_lr,
                       weight_decay=args.weight_decay,
                       eps=args.adam_epsilon)


def to_tensor_float(data):
    return torch.as_tensor(data, dtype=torch.float)


def to_tensor_long(data):
    return torch.as_tensor(data, dtype=torch.long)


def get_tensor_info(tensor):
    return f"Shape: {tensor.shape} | Type: {tensor.type()} | Device: {tensor.device}"


def set_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.n_gpu > 0 and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
