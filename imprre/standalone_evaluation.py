import argparse

import numpy as np
import torch

import logging
import logging.config
import os.path

from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

from evaluation import get_f1
from model import REModel
from prepro import RETACREDProcessor
from utils import collate_fn
from utils_imported import scorer

log = logging.getLogger(__name__)


#
#
# def load_state_dict(self, ckpt):
#     checkpoint = torch.load(ckpt)
#     self.model.load_state_dict(checkpoint['state_dict'])
#
#
# def standalone_eval():
#     model = model
#     load_state_dict(config.kpwr_ckpt)
#     eval_kpwr(framework.test_loader)
#



def do_eval(args, model, benchmarks, processor):

    print ("evaluating ...")
    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, processor,  tag=tag)



def evaluate(args, model, features, processor,  tag='dev'):

    print('Starting evaluation '+tag)
    log.debug("Starting evaluation "+tag)

    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn_eval, drop_last=False)
    keys, preds, negative_samples = [], [], []
    sentences = []

    # wersja KPWr
    LABEL_TO_ID =      {'no_relation': 0,
                        'affiliation': 1,
                        'alias': 2,
                        'composition': 3,
                        'creator': 4,
                        'location': 5,
                        'nationality': 6,
                        'neighbourhood': 7,
                        'origin': 8}

    id2label = dict([(v, k) for k, v in LABEL_TO_ID.items()])



    prec_micro, recall_micro = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()
        current_sentences = []

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'ss': batch[3].to(args.device),
                  'os': batch[4].to(args.device),
                  }

        current_keys = batch[2].tolist()
        keys += current_keys
        tokenids = batch[0].tolist()
        #print("tokenids ="+str(tokenids))

        for tkids in tokenids:
            #print ("processing tkids="+str(tkids))
            sent =  processor.tokenizer.convert_ids_to_tokens(tkids)
            # print("1"+str(sent))
            sent = ''.join(sent)
            # print("2" + str(sent))
            sent = sent.replace('</w>',' ')
            sent = sent.replace('<s>', ' ')
            sent = sent.replace('</s>',' ')
            sent = sent.replace('#','# ')
            sent = sent.replace('@', '@ ')
            # print ("3 sent ="+str(sent))
            current_sentences.append(sent)

        with torch.no_grad():
            logit = model(**inputs)[0]
            current_preds = torch.argmax(logit, dim=-1)
        current_preds = current_preds.tolist();
        preds += current_preds

        sample_ids = batch[5]

        current_negative_sample  = [ (sample_ids[index], current_keys[index],
                                      current_preds[index], current_sentences[index])
                                       for index in range(len(sample_ids)) 
                                       if current_keys[index] != current_preds[index]
                                      ]

        negative_samples += current_negative_sample
        sentences.append(current_sentences)


    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    prec_micro, recall_micro, max_f1 = get_f1(keys, preds)


    predictions = [id2label[p] for p in preds]
    gold_tot = [id2label[p] for p in keys]


    results_file = open(tag+"_kpwr_results_file_negative.csv","w");
    results_file.write("ID ; GOLD ; PRED ; subject_type; object_type ; zdanie \n")
    for id, key, pred, sent in negative_samples:
        ner_type_1, ner_type_2= extract_ner_types(sent)
        results_file.write("NEG_ID ="+str(id)+";"+id2label[key]+";"+id2label[pred]+";"+ner_type_1+";"+ner_type_2+ ";"+str(sent)+"\n")
        #results_file.write("N_ID =" + str(id) + ";" + id2label[key] + ";" + id2label[pred] +  "\n")

    results_file.close()


    print("====scorer====");
    scorer.score(gold_tot, predictions, verbose=True)
    print("====scorer====");

    output = {
        tag + "_f1": max_f1 * 100,
        tag + "_pr": prec_micro * 100,
        tag + "_re": recall_micro * 100,
    }
    print(output )
    return max_f1, output



def collate_fn_eval(batch):
    max_len = max([len(f["input_ids"]) for f in batch])
    input_ids = [f["input_ids"] + [0] * (max_len - len(f["input_ids"])) for f in batch]
    input_mask = [[1.0] * len(f["input_ids"]) + [0.0] * (max_len - len(f["input_ids"])) for f in batch]
    labels = [f["labels"] for f in batch]
    ss = [f["ss"] for f in batch]
    os = [f["os"] for f in batch]
    input_ids = torch.tensor(input_ids, dtype=torch.long)
    input_mask = torch.tensor(input_mask, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)
    ss = torch.tensor(ss, dtype=torch.long)
    os = torch.tensor(os, dtype=torch.long)
    sample_ids = [f["sample_ids"] for f in batch]
    output = (input_ids, input_mask, labels, ss, os, sample_ids)
    return output

def extract_ner_types(sent):
    start_1 = sent.find('*')+1
    end_1 = sent.find('*',start_1)

    start_2 = sent.find('^')+1
    end_2 = sent.find('^',start_2)

    #print ("S1,E1|S2,E2 = "+str(start_1)+","+str(end_1)+" | "+str(start_2)+","+str(end_2))

    return sent[start_1:end_1], sent[start_2:end_2]


def main():

    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
            },
        },
        'handlers': {
            'default_handler': {
                'class': 'logging.FileHandler',
                'level': 'ERROR',
                'formatter': 'standard',
                'filename': os.path.join('logs', 'application.log'),
                'encoding': 'utf8'
            },
            'console_handler': {
                'class': 'logging.StreamHandler',
                'level': 'ERROR',
                'formatter': 'standard',
            },

        },
        'loggers': {
            '': {
                'handlers': ['default_handler', 'console_handler'],
                'level': 'ERROR',
                'propagate': False
            }
        }
    }
    logging.config.dictConfig(logging_config)

    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", default="./data/retacred", type=str)
    parser.add_argument("--model_name_or_path", default="roberta-large", type=str)
    parser.add_argument("--input_format", default="typed_entity_marker_punct", type=str,
                        help="in [entity_mask, entity_marker, entity_marker_punct, typed_entity_marker, typed_entity_marker_punct]")

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated.")

    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=32, type=int,
                        help="Batch size for testing.")


    # parser.add_argument("--train_batch_size", default=8, type=int,
    #                     help="Batch size for training.")
    # parser.add_argument("--test_batch_size", default=8, type=int,
    #                     help="Batch size for testing.")

    # used for debugging
    # parser.add_argument("--train_batch_size", default=2, type=int,
    #                     help="Batch size for training.")
    # parser.add_argument("--test_batch_size", default=2, type=int,
    #                     help="Batch size for testing.")



    parser.add_argument("--learning_rate", default=3e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--gradient_accumulation_steps", default=2, type=int,
                        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument("--warmup_ratio", default=0.1, type=float,
                        help="Warm up ratio for Adam.")

    # used for debugging
    # parser.add_argument("--num_train_epochs", default=1.0, type=float,
    #                   help = "Total number of training epochs to perform.")

    parser.add_argument("--num_train_epochs", default=20.0, type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="random seed for initialization")
    parser.add_argument("--num_class", type=int, default=40)

    parser.add_argument("--seed", type=int, default=9,
                        help="random seed for initialization")
    #parser.add_argument("--num_class", type=int, default=9)

    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="re-tacred")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    config = AutoConfig.from_pretrained(
        args.config_name if args.config_name else args.model_name_or_path,
        num_labels=args.num_class,
    )
    config.gradient_checkpointing = True
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
    )

    model = REModel(args, config)
    model.to(0)

    #train_file = os.path.join(args.data_dir, "train.json")
    dev_file = os.path.join(args.data_dir, "dev.json")
    test_file = os.path.join(args.data_dir, "test.json")

    processor = RETACREDProcessor(args, tokenizer)
    #train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        #("train", train_features),  # i tak 100%
        ("dev", dev_features),
        ("test", test_features),
    )

    print ("loading model ...")
    checkpoint = torch.load("checkpoint/imprre.pth.tar")
    model.load_state_dict(checkpoint['state_dict'])

    print('Starting ...')
    log.debug("Starting .....")
    do_eval(args, model, benchmarks,processor)


if __name__ == "__main__":
    main()
