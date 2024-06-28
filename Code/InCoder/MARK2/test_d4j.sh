beam_size=10
output_size=$beam_size
batch_size=1
model=Epoch_1
input_dir=/Data/Defects4J_dataset/Transfer-D4j/Mark2/V1
# input_dir=/data3/HuangKai/Dataset/Vul4J_dataset/single_hunk_vul/Mark2
output_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_InCoder-6B/model_set_mark2_2048_LORA/D4J/beam_size_$beam_size/more
model_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_InCoder-6B/model_set_mark2_2048_LORA

mkdir -p $output_dir

python test_d4j.py \
        --model_name_or_path $model_dir/$model/-merged \
        --test_filename $input_dir/src-test.jsonl,$input_dir/tgt-test.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log