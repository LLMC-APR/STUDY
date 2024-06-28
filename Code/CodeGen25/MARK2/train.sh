lr=5e-5
beam_size=1
epoch=1
batch_size=1
model=Salesforce/codegen25-7b-multi
input_dir=/Data/Transfer_dataset/2-Mark2
output_dir=/Data/Transfer_dataset/Result/1_LLM4APR/result_CodeGen25-7B/model_set_mark2_2048_LORA

# Need old version transformers

mkdir -p $output_dir

python mark2.py \
        --model_name_or_path $model \
        --train_filename $input_dir/src-train.jsonl,$input_dir/tgt-train.jsonl \
        --dev_filename $input_dir/src-val.jsonl,$input_dir/tgt-val.jsonl \
        --output_dir $output_dir \
        --max_source_length 2048 \
        --max_target_length 256 \
        --beam_size $beam_size \
        --train_batch_size $batch_size \
        --learning_rate $lr \
        --num_train_epochs $epoch \
        --eval_step 1 \
        2>&1 | tee $output_dir/train.log