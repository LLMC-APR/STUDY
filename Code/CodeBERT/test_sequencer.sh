data_dir=/SS970evo/datasets/SequenceR_dataset/preprocess/mark2_src
output_dir=/SS970evo/datasets/SequenceR_dataset/result_CodeBERT/mark2_src/model_set1
test_model_type=best-bleu #last best-bleu/ppl

mkdir -p $output_dir

python run_sequencer.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $output_dir/checkpoint-$test_model_type/pytorch_model.bin \
        --test_filename $data_dir/src-test.txt,$data_dir/tgt-test.txt \
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 128 \
        --beam_size 50 \
        --eval_batch_size 16 \
        --learning_rate 5e-5 \
        2>&1 | tee $output_dir/test_$test_model_type.log