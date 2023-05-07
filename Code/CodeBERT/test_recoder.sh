data_dir=/SS970evo/datasets/Defects4J_dataset/mark2_D4j_v1
model_dir=/SS970evo/datasets/Recoder_dataset/result_CodeBERT/mark2/model_set1
output_dir=$model_dir/D4J_note
test_model_type=last #last best-bleu/ppl

mkdir -p $output_dir

python run_sequencer.py \
        --do_test \
        --model_type roberta \
        --model_name_or_path microsoft/codebert-base \
        --load_model_path $model_dir/checkpoint-$test_model_type/pytorch_model.bin \
        --test_filename $data_dir/note-src-test.txt,$data_dir/note-tgt-test.txt \
        --output_dir $output_dir \
        --max_source_length 512 \
        --max_target_length 512 \
        --beam_size 100 \
        --eval_batch_size 1 \
        --learning_rate 5e-4 \
        2>&1 | tee $output_dir/test_$test_model_type.log