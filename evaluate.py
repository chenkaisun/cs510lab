from torch.utils.data import DataLoader
import torch
import torch.utils.data
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error, \
    precision_score, recall_score
import numpy as np
from data import collate_wrapper


def get_prf(targets, preds, average="micro", verbose=False):
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
            pred = model(inputs, args)
            pred = list(pred.cpu().numpy())
            preds.extend(pred)
            ids.extend(batch.ids)
            if args.exp == "mol_pred":
                targets.extend(list(inputs.batch_graph_data.y.cpu().numpy()))
            else:
                targets.extend(list(inputs.labels.cpu().numpy()))
    preds = np.array(preds)


    precision, recall, score = get_prf(targets, preds.tolist())

    return score, [list(item) for item in zip(ids, preds.tolist(), targets)]
