from torch.utils.data import DataLoader
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score
import numpy as np
from data import collate_wrapper


def get_prf(targets, preds, average="samples", verbose=False):
    precision = precision_score(targets, preds, average=average)
    recall = recall_score(targets, preds, average=average)
    f1 = f1_score(targets, preds, average=average)
    # print(precision, recall, f1)
    if verbose: print(f"{average}: precision {precision} recall {recall} f1 {f1}")
    return precision, recall, f1


def evaluate(args, model, data):

    dataloader = DataLoader(data, batch_size=args.batch_size, shuffle=False, collate_fn=collate_wrapper,
                            drop_last=False)
    preds = []
    targets = []
    ids = []
    for batch in dataloader:
        model.eval()
        batch.in_train = False
        inputs = batch.to(args.device)

        # todo: change evaluation
        with torch.no_grad():
            ids.extend(batch.ids)
            pred = model(inputs, args)
            preds.extend(list(pred.cpu().numpy()))
            targets.extend(list(inputs.labels.cpu().numpy()))
    # preds = np.array(preds) .tolist() .tolist()
    precision, recall, score = get_prf(targets, preds, average="samples")

    return score, [list(item) for item in zip(ids, preds, targets)]
