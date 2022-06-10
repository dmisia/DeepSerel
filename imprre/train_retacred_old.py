import argparse
import os

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer
from transformers.optimization import AdamW, get_linear_schedule_with_warmup
from utils import set_seed, collate_fn
from prepro import RETACREDProcessor
from evaluation import get_f1
from model import REModel
from torch.cuda.amp import GradScaler
from utils_imported import scorer
import wandb

import logging
import logging.config
import os.path

log = logging.getLogger(__name__)



def train(args, model, train_features, benchmarks):
    train_dataloader = DataLoader(train_features, batch_size=args.train_batch_size, shuffle=True, collate_fn=collate_fn, drop_last=True)
    total_steps = int(len(train_dataloader) * args.num_train_epochs // args.gradient_accumulation_steps)
    warmup_steps = int(total_steps * args.warmup_ratio)

    scaler = GradScaler()
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)
    print('Total steps: {}'.format(total_steps))
    print('Warmup steps: {}'.format(warmup_steps))

    best_f1=0
    num_steps = 0
    for epoch in range(int(args.num_train_epochs)):
        torch.cuda.empty_cache()
        model.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader)):

            log.debug("step="+str(step))
            log.debug("batch="+str(batch))

            torch.cuda.empty_cache()
            model.train()
            inputs = {'input_ids': batch[0].to(args.device),
                      'attention_mask': batch[1].to(args.device),
                      'labels': batch[2].to(args.device),
                      'i_s': batch[3].to(args.device),  # indeksy subjectów w stokenizowanym już wejściu, tylko początki
                      't_s': batch[4].to(args.device),  # indeksy objectów w stokenizowanym już wejściu, tylko początki
                      'l_s': batch[5].to(args.device),  # indeksy objectów w stokenizowanym już wejściu, tylko początki
                      }

            log.debug("inputs = "+str(inputs))

            outputs = model(**inputs)

            # log.debug("outputs = "+str(outputs))
            # log.debug("args.gradient_accumulation_steps = "+str(args.gradient_accumulation_steps))
            # log.debug("outputs[0] ="+str(outputs[0]))

            loss = outputs[0] / args.gradient_accumulation_steps

            # log.debug("loss = outputs[0]/args.gradient_accumulation_steps = "+str(loss))
            torch.cuda.empty_cache()
            scaler.scale(loss).backward()

            # ta część najprawdopodobniej po to by obejść ograniczenia pamięci karty graficznej
            if step % args.gradient_accumulation_steps == 0:
                torch.cuda.empty_cache()
                num_steps += 1
                if args.max_grad_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                model.zero_grad()
                wandb.log({'loss': loss.item()}, step=num_steps)

            if (num_steps % args.evaluation_steps == 0 and step % args.gradient_accumulation_steps == 0):
                for tag, features in benchmarks:
                    print("Evaluation during trening, epoch: " + str(epoch))
                    f1, output = evaluate(args, model, features, tag=tag)
                    wandb.log(output, step=num_steps)
                    if ( f1 > best_f1 and tag == 'dev' ) :
                        print("Best ckpt and saved.")
                        torch.save({'state_dict': model.state_dict()}, "checkpoint/imprre.pth.tar")
                        best_f1 = f1


    print("Evaluation after trening")
    for tag, features in benchmarks:
        f1, output = evaluate(args, model, features, tag=tag)

        wandb.log(output, step=num_steps)


def evaluate(args, model, features, tag='dev'):
    dataloader = DataLoader(features, batch_size=args.test_batch_size, collate_fn=collate_fn, drop_last=False)
    keys, preds = [], []
    prec_micro, recall_micro = [], []
    for i_b, batch in enumerate(dataloader):
        model.eval()

        inputs = {'input_ids': batch[0].to(args.device),
                  'attention_mask': batch[1].to(args.device),
                  'i_s': batch[3].to(args.device),
                  't_s': batch[4].to(args.device),
                  'l_s': batch[5].to(args.device)
                  }
        keys += batch[2].tolist()
        with torch.no_grad():
            logit = model(**inputs)[0]
            pred = torch.argmax(logit, dim=-1)
        preds += pred.tolist()

    keys = np.array(keys, dtype=np.int64)
    preds = np.array(preds, dtype=np.int64)
    prec_micro, recall_micro, max_f1 = get_f1(keys, preds)


    # dodane

    LABEL_TO_ID = {'no_relation': 0,
                   'relation': 1}

    id2label = dict([(v, k) for k, v in LABEL_TO_ID.items()])
    predictions = [id2label[p] for p in preds]
    gold_tot = [id2label[p] for p in keys]

    print("====scorer====");
    scorer.score(gold_tot,predictions,verbose=True)
    print("====scorer====");

    output = {
        tag + "_f1": max_f1 * 100,
        tag + "_pr": prec_micro * 100,
        tag + "_re": recall_micro * 100,
    }
    print(output )
    return max_f1, output


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
                'level': 'INFO',
                'formatter': 'standard',
                'filename': os.path.join('logs', 'application.log'),
                'encoding': 'utf8'
            },
            'console_handler': {
                'class': 'logging.StreamHandler',
                'level': 'INFO',
                'formatter': 'standard',
            },

        },
        'loggers': {
            '': {
                'handlers': ['default_handler', 'console_handler'],
                'level': 'INFO', #DEBUG
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
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size for training.")
    parser.add_argument("--test_batch_size", default=8, type=int,
                        help="Batch size for testing.")

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

    parser.add_argument("--num_train_epochs", default=10.0, type=float,
                        help="Total number of training epochs to perform.")
    # parser.add_argument("--seed", type=int, default=42,
    #                     help="random seed for initialization")
    # parser.add_argument("--num_class", type=int, default=40)
    #
    parser.add_argument("--seed", type=int, default=9,
                        help="random seed for initialization")
    #dwie klasy -> relation lub no_relation
    parser.add_argument("--num_class", type=int, default=2)

    parser.add_argument("--evaluation_steps", type=int, default=500,
                        help="Number of steps to evaluate the model")

    parser.add_argument("--dropout_prob", type=float, default=0.1)
    parser.add_argument("--project_name", type=str, default="RE_baseline")
    parser.add_argument("--run_name", type=str, default="re-tacred")

    args = parser.parse_args()
    wandb.init(project=args.project_name, name=args.run_name)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    if args.seed > 0:
        set_seed(args)

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

    # train_file = os.path.join(args.data_dir, "train_min.json")
    # dev_file = os.path.join(args.data_dir, "dev_min.json")
    # test_file = os.path.join(args.data_dir, "test_min.json")

    train_file = os.path.join(args.data_dir, "train_triples.json")
    dev_file = os.path.join(args.data_dir, "dev_triples.json")
    test_file = os.path.join(args.data_dir, "test_triples.json")

    processor = RETACREDProcessor(args, tokenizer)
    train_features = processor.read(train_file)
    dev_features = processor.read(dev_file)
    test_features = processor.read(test_file)

    if len(processor.new_tokens) > 0:
        model.encoder.resize_token_embeddings(len(tokenizer))

    benchmarks = (
        ("dev", dev_features),
        ("test", test_features),
    )

    print('Starting train ...')
    log.debug("Starting train .....")
    train(args, model, train_features, benchmarks)


if __name__ == "__main__":
    main()
