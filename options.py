import argparse


def read_args():
    parser = argparse.ArgumentParser()

    # pretrained language model
    parser.add_argument("--plm", default="bert-base-cased", type=str, metavar='N')
    parser.add_argument("--max_seq_len", default=1024, type=int)

    # experiment
    parser.add_argument("--model_name", default="baseline", type=str)
    parser.add_argument("--model_path", default="model/states/best_dev.pt", type=str)
    parser.add_argument("--experiment_path", default="experiment/", type=str)
    parser.add_argument("--exp", default="bio", type=str)
    parser.add_argument("--analyze", default=0, type=int)
    parser.add_argument("--debug", default=1, type=int, help="Using gpu or cpu", )
    parser.add_argument("--eval", action="store_true")

    # data
    parser.add_argument("--seed", type=int, default=0, help="random seed for initialization")
    parser.add_argument("--tgt_name", default="p_np", type=str)
    parser.add_argument("--train_file", default="data/property_pred/clintox.csv", type=str)
    parser.add_argument("--val_file", default="dev.json", type=str)
    parser.add_argument("--test_file", default="test.json", type=str)
    parser.add_argument("--cache_filename", default="data/saved.pkl", type=str)
    parser.add_argument("--use_cache", default=0, type=int)
    parser.add_argument("--num_workers", default=1, type=int)

    # training params
    parser.add_argument("--batch_size", default=2, type=int, help="Batch size for training.")
    parser.add_argument("--num_epochs", default=30, type=float, help="Total number of training epochs to perform.")
    # parser.add_argument("--eval_epoch", default=30, type=float, help="Number of steps between each evaluation.")
    # parser.add_argument('--print_epoch_interval', type=int, default=10)
    parser.add_argument('--burn_in', type=int, default=0)
    parser.add_argument('--patience', type=int, default=8)

    parser.add_argument("--plm_lr", default=2e-5, type=float, help="The initial learning rate for PLM.")
    parser.add_argument("--lr", default=1e-4, type=float, help="The initial learning rate for Adam.")

    parser.add_argument('--weight_decay', type=float, default=5e-4, help='Weight decay (L2 loss on parameters).')
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")

    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--grad_accumulation_steps", default=1, type=int, help="Using mixed precision")

    parser.add_argument("--scheduler", default="linear")
    parser.add_argument("--warmup_ratio", default=0.06, type=float, help="Warm up ratio for Adam.")

    parser.add_argument("--paralell", action="store_true", help="Use paralell multiple gpu")
    parser.add_argument("--n_gpu", default=1, type=int, help="Number of gpu", )
    parser.add_argument("--use_gpu", default=1, type=int, help="Using gpu or cpu", )
    parser.add_argument("--use_amp", default=1, type=int, help="Using mixed precision")

    # model params
    parser.add_argument("--model_type", default="", type=str, help="model_type")

    parser.add_argument("--in_dim", default=14, type=float, help="Feature dim")
    parser.add_argument("--out_dim", default=14, type=float, help="Feature dim")
    parser.add_argument('--batch_norm', default=False, help="Please give a value for batch_norm")
    parser.add_argument("--dropout", default=0.1, type=float, help="Dropout")


    parser.add_argument('--g_dim', type=int, default=256, help='Number of final hidden units for graph.')
    parser.add_argument('--num_gnn_layers', type=int, default=3, help='Number of final hidden units for graph.')

    parser.add_argument('--plm_hidden_dim', type=int, default=128, help='Number of hidden units for plm.')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Number of hidden units.')
    parser.add_argument('--embedding_dim', type=int, default=16, help='Number of embedding units.')

    args = parser.parse_args()
    return args
