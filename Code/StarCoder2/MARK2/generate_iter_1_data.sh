beam_size=1
output_size=$beam_size
batch_size=1
model=Epoch_1
input_dir=/mnt/share/huangk/Dataset/Transfer_dataset/2-Mark2
# input_dir=/data3/HuangKai/Dataset/Vul4J_dataset/single_hunk_vul/Mark2
output_dir=/mnt/share/huangk/Dataset/Transfer_dataset/2-ITER/beam_size_$beam_size
model_dir=/mnt/share/huangk/Dataset/Transfer_dataset/RQ2/result_StarCoder2-15B/model_set_mark2_2048_LORA

mkdir -p $output_dir

python generate_iter.py \
        --model_name_or_path $model_dir/$model/-merged \
        --test_filename $input_dir/src-train.jsonl,$input_dir/tgt-train.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --output_size $output_size \
        --test_batch_size $batch_size \
        2>&1 | tee $output_dir/test.log