#for SEED in 78 23 61;
for SEED in 23 61;
#do python3 train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
#do python3 train_retacred.py --model_name_or_path /home/michalolek/NLPWR/embeddings/RoBERTa/roberta_large_transformers --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;
#do python3 train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta;

#polskie zanurzenia
#do python3 train_retacred.py --model_name_or_path ./roberta_base_transformers --input_format typed_entity_marker --seed $SEED --run_name roberta --num_class 9;

#polskie z zanuerzanimi roberta_kgr10
#do python3 train_retacred.py --model_name_or_path clarin-pl/roberta-polish-kgr10 --input_format typed_entity_marker_punct --seed $SEED --run_name roberta --num_class 9;

#polskie z zanuerzanimi herberta
do python3 train_retacred.py --model_name_or_path allegro/herbert-large-cased --input_format typed_entity_marker_punct --seed $SEED --run_name herbertInq --num_class 9;


#angielskie zanurzenia
#do python3 train_retacred.py --model_name_or_path roberta-large --input_format typed_entity_marker_punct --seed $SEED --run_name roberta ;

done;
