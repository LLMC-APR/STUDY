
data_dir=/SS970evo/datasets/Defects4J_dataset/mark2_D4j_short_v2
model_dir=/SS970evo/datasets/CPatMiner_dataset/result_PLBART/mark2/model_set1
output_dir=$model_dir
checkpoint_model_type=best-bleu  # best-bleu/ppl or last
log_file=test_$checkpoint_model_type.log

max_source_length=512
max_target_length=256
beam_size=100
eval_batch_size=1
learning_rate=5e-5

python run_sequencer.py \
        --do_test \
        --model_type bart \
        --model_name_or_path ../../../PLBART/pretrain/checkpoint_11_100000.pt \
        --tokenizer_name ../../../PLBART/sentencepiece/sentencepiece.bpe.model \
        --load_model_path $model_dir/checkpoint-$checkpoint_model_type/pytorch_model.bin \
        --test_filename $data_dir/src-test.txt,$data_dir/tgt-test.txt \
        --output_dir $output_dir \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --beam_size $beam_size \
        --eval_batch_size $eval_batch_size \
        --learning_rate $learning_rate \
        --checkpoint_model_type $checkpoint_model_type \
        2>&1 | tee $output_dir/$log_file