import torch.nn as nn


class CrossEncoder(nn.Module):

    def __init__(self, pretrained_model):
        super().__init__()
        self.transformer = pretrained_model
        self.regressor = nn.Linear(self.transformer.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.transformer(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        cls_emb = out.hidden_states[-1][:, 0, :]
        return self.regressor(cls_emb)
