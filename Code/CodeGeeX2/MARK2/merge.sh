# beam_size=1
# output_size=1
# input_dir=/data3/HuangKai/Dataset/TRANSFER_dataset/template_sec
output_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_CodeGeeX2-6B/model_set_mark2_2048_LORA

# mkdir -p $output_dir

python merge_peft_adapters.py \
        --base_model_name_or_path THUDM/codegeex2-6b \
        --peft_model_path $output_dir/Epoch_1/  \
        --push_to_hub \