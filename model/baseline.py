import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from transformers import AutoModel
from model.model_utils import get_tensor_info, LabelSmoothingCrossEntropy


class baseline(nn.Module):
    def __init__(self, args):
        super().__init__()
        # self.model_type = args.model_type
        self.plm = AutoModel.from_pretrained(args.plm)
        # self.criterion = LabelSmoothingCrossEntropy(reduction='sum')
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.dropout = args.dropout
        # self.loss = torch.nn.CrossEntropyLoss()
        self.final = nn.Linear(args.plm_hidden_dim, args.out_dim)

    def forward(self, input,args):
        token_ids = input.token_ids
        in_train = input.in_train
        labels = input.labels

        # print(token_ids)

        hid_texts = self.plm(**token_ids, return_dict=True).pooler_output
        # print("hid_texts", get_tensor_info(hid_texts))
        output = self.final(hid_texts)

        # todo: change loss and prediction depending on multi-label/multi-class etc
        if in_train:
            # label smoothing
            return self.criterion(output, labels)
            # return self.loss(output, labels)
            # return torch.nn.functional.cross_entropy(output, labels)
            # return torch.nn.functional.binary_cross_entropy_with_logits(output, labels)
        # return torch.argmax(F.log_softmax(output, dim=-1), dim=-1)

        pred_out = (torch.sigmoid(output) > 0.5).float()

        return pred_out
