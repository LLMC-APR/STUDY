beam_size=10
output_size=$beam_size
batch_size=1
# top_n=10
max_input_size=2048
model_name=CodeLlama-13b-hf # CodeLlama-34b-hf
input_dir=/Data/Defects4J_dataset/Transfer-D4j/Mark2/V1
# input_dir=/data3/HuangKai/Dataset/Recoder_dataset/2-Program_Repair/Recoder_test/$ts_model/top_$top_n
model_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_CodeLlama-13B/model_set_mark2_2048_LORA
output_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_CodeLlama-13B/model_set_mark2_2048_LORA/D4J/beam_size_$beam_size

mkdir -p $output_dir

python test_d4j.py \
        --model_name_or_path $model_dir/Epoch_1/-merged \
        --test_filename $input_dir/src-test.jsonl,$input_dir/tgt-test.jsonl \
        --output_dir $output_dir \
        --max_source_length $max_input_size \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log