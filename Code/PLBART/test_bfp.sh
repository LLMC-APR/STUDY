
data_dir=/SS970evo/datasets/Tufano_dataset/datasets/preprocess/50/ori_abs
output_dir=/SS970evo/datasets/Tufano_dataset/datasets/result_PLBART/50/ori_abs/model_set1
checkpoint_model_type=last  # best-bleu/ppl or last
log_file=test_$checkpoint_model_type.log

max_source_length=512
max_target_length=512
beam_size=5
eval_batch_size=8
learning_rate=5e-5

python run.py \
        --do_test \
        --model_type bart \
        --model_name_or_path ../../../PLBART/pretrain/checkpoint_11_100000.pt \
        --tokenizer_name ../../../PLBART/sentencepiece/sentencepiece.bpe.model \
        --test_filename $data_dir/src-test.txt,$data_dir/tgt-test.txt \
        --output_dir $output_dir \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --beam_size $beam_size \
        --eval_batch_size $eval_batch_size \
        --learning_rate $learning_rate \
        --checkpoint_model_type $checkpoint_model_type \
        2>&1 | tee $output_dir/$log_file