from train import train
from data import OurDataset
from train_utils import *

from utils import dump_file, load_file, get_tokenizer
from evaluate import evaluate
from options import *

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = read_args()
    set_seeds(args)

    # todo: preset everything in options.py that stay the same for this experiment, like below
    data_dir = "/data/jinning/mesh/"
    train_file = data_dir + "allMeSH_2021.json"
    val_file = data_dir + "dev.txt"
    test_file = data_dir + "test.txt"
    args.model_name = "baseline"
    args.exp = "bioasp_task1_basebert"
    args.plm = "allenai/scibert_scivocab_uncased"

    # can run on local computer
    if args.debug:
        print("debug")
        args.plm = "prajjwal1/bert-tiny"
        args.batch_size=2
        args.num_epochs=2
    print("PLM is", args.plm)

    tokenizer = get_tokenizer(args.plm)
    print("Tokenizer loaded")

    train_data = OurDataset(args, train_file, tokenizer)
    val_data, test_data = train_data, train_data

    # set up optimizer and model, move to gpu, set up loggers
    args, model, optimizer = setup_common(args)

    if args.analyze:
        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        test_score, output = evaluate(args, model, test_data)

    else:
        train(args, model, optimizer, (train_data, val_data, test_data))
