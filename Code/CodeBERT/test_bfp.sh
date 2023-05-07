data_dir=/SS970evo/datasets/Tufano_dataset/datasets/preprocess/50/ori_src
output_dir=/SS970evo/datasets/Tufano_dataset/datasets/result_CodeBERT/50/ori_src/model_set1
test_model_type=last #last best-bleu/ppl

mkdir -p $output_dir

python run.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $output_dir/checkpoint-$test_model_type/pytorch_model.bin \
        --test_filename $data_dir/src-test.txt,$data_dir/tgt-test.txt \
        --output_dir $output_dir \
        --max_source_length 256 \
        --max_target_length 256 \
        --beam_size 5 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        2>&1 | tee $output_dir/test20230426_$test_model_type.log