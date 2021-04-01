from train import train
from data import OurDataset
from train_utils import *

from utils import dump_file, load_file, get_tokenizer
from evaluate import evaluate


torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

if __name__ == '__main__':
    args = read_args()
    data_dir = "data/"
    train_file = data_dir + "train.txt"
    val_file = data_dir + "dev.txt"
    test_file = data_dir + "test.txt"
    args.model_name = "baseline"
    args.exp = ""
    args.plm = "allenai/scibert_scivocab_uncased"
    if args.debug:
        args.plm = "prajjwal1/bert-tiny"

    print("PLM is",args.plm)
    set_seeds(args)

    tokenizer = get_tokenizer(args.plm)
    print("Tokenizer loaded")

    train_data, val_data, test_data = OurDataset(args, train_file, tokenizer), \
                                      OurDataset(args, val_file, tokenizer), \
                                      OurDataset(args, test_file, tokenizer)

    # set up optimizer and model, move to gpu, set up logger
    args, model, optimizer = setup_common(args)
    if args.analyze:
        model.load_state_dict(torch.load(args.model_path)['model_state_dict'])
        test_score, output = evaluate(args, model, test_data)

    else:
        train(args, model, optimizer, (train_data, val_data, test_data))
