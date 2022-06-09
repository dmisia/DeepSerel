import torch
import torch.nn as nn
from transformers import AutoModel
from torch.cuda.amp import autocast

import logging
import logging.config
import os.path

log = logging.getLogger(__name__)





class REModel(nn.Module):
    def __init__(self, args, config):
        super().__init__()
        self.args = args
        self.encoder = AutoModel.from_pretrained(args.model_name_or_path, config=config)
        hidden_size = config.hidden_size
        self.loss_fnt = nn.CrossEntropyLoss()
        self.classifier = nn.Sequential(
            nn.Linear(3 * hidden_size, hidden_size), # dodany jeden argument
            nn.ReLU(),
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(hidden_size, args.num_class)
        )

    @autocast()
    def forward(self, input_ids=None, attention_mask=None, labels=None, i_s=None, t_s=None, l_s=None):
        outputs = self.encoder(
            input_ids,
            attention_mask=attention_mask,
        )
        log.debug("outputs just from encoder = "+str(outputs))
        log.debug("outputs just from encoder keys = " + str(outputs.keys()))


        pooled_output = outputs[0]

        log.debug("pooled_output= "+str(pooled_output))

        idx = torch.arange(input_ids.size(0)).to(input_ids.device)

        log.debug("idx= " + str(idx))

        i_s_emb = pooled_output[idx, i_s]
        log.debug("i_s_emb = " + str(i_s_emb))

        t_s_emb = pooled_output[idx, t_s]
        log.debug("t_s_emb = " + str(t_s_emb))

        l_s_emb = pooled_output[idx, l_s]
        log.debug("l_s_emb = " + str(l_s_emb))

        h = torch.cat((i_s_emb, t_s_emb, l_s_emb), dim=-1)
        log.debug("h = " + str(h))


        logits = self.classifier(h)
        log.debug("logits = "+str(logits));


        outputs = (logits,)
        log.debug("outputs z (logits,) = " + str(outputs));

        log.debug("---------------------")
        log.debug("logits.float() = " + str(logits.float()));
        log.debug("labels = " + str(labels));
        log.debug("---------------------")

        # dla predykcji jest None, dla uczenia nie jest None
        if labels is not None:
            log.debug("labels is not None");

            loss = self.loss_fnt(logits.float(), labels)
            log.debug("loss ="+str(loss));

            outputs = (loss,) + outputs
            log.debug("outputs concatenated =" + str(outputs));

        log.debug("returning outputs :"+str(outputs))
        return outputs
