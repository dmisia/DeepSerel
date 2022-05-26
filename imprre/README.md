# RE_improved_baseline

```
REZULTATY DLA KPWR:

Per-relation statistics:
affiliation    P:  75.66%  R:  73.25%  F1:  74.43%  gold:  157  correct:  115  shots:  152
alias          P:  64.79%  R:  76.67%  F1:  70.23%  gold:   60  correct:   46  shots:   71
composition    P:  83.22%  R:  87.32%  F1:  85.22%  gold:  142  correct:  124  shots:  149
creator        P:  60.34%  R:  60.34%  F1:  60.34%  gold:   58  correct:   35  shots:   58
location       P:  74.33%  R:  75.91%  F1:  75.11%  gold:  328  correct:  249  shots:  335
nationality    P:  71.05%  R:  72.97%  F1:  72.00%  gold:   37  correct:   27  shots:   38
neighbourhood  P:  40.00%  R:  41.86%  F1:  40.91%  gold:   43  correct:   18  shots:   45
origin         P:  82.76%  R:  64.86%  F1:  72.73%  gold:   37  correct:   24  shots:   29

Final Score:
Precision (micro): 72.748%
   Recall (micro): 74.013%
       F1 (micro): 73.375%
```



Code for technical report "[An Improved Baseline for Sentence-level Relation Extraction](https://arxiv.org/abs/2102.01373)".

## Requirements
* torch >= 1.8.1
* transformers >= 3.4.0
* wandb
* ujson
* tqdm

The Pytorch version must be at least 1.8.1 as our code relies on the both the ``torch.cuda.amp`` and the ``torch.utils.checkpoint``, which are introduced in the 1.8.1 release.

## Dataset
The TACRED dataset can be obtained from [this link](https://nlp.stanford.edu/projects/tacred/). The TACREV and Re-TACRED dataset can be obtained following the instructions in [tacrev](https://github.com/DFKI-NLP/tacrev) and [Re-TACRED](https://github.com/gstoica27/Re-TACRED), respectively. The expected structure of files is:
```
RE_improved_baseline
 |-- dataset
 |    |-- tacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
 |    |    |-- dev_rev.json
 |    |    |-- test_rev.json
 |    |-- retacred
 |    |    |-- train.json        
 |    |    |-- dev.json
 |    |    |-- test.json
```

## Training and Evaluation
The commands and hyper-parameters for running experiments can be found in the ``scripts`` folder. For example, to train roberta-large, run
```bash
>> sh run_roberta_tacred.sh    # TACRED and TACREV
>> sh run_roberta_retacred.sh  # Re-TACRED
```
The evaluation results are synced to the wandb dashboard. The results on TACRED and TACREV can be obtained in one run as they share the same training set.

For all tested pre-trained language models, training can be conducted with one RTX 2080 Ti card.
