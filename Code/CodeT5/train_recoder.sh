
lr=5e-5
batch_size=16 #32
beam_size=5
max_source_length=512
max_target_length=256
epoch=30

input_dir=/SS970evo/datasets/Recoder_dataset/mark2
output_dir=/SS970evo/datasets/Recoder_dataset/result_CodeT5/mark2/model_set1
log_file=train.log
# model_dir = $output_dir/save_checkpoints
res_dir=$output_dir
summary_dir=$output_dir/summary_data
train_file=$input_dir/src-train.txt,$input_dir/tgt-train.txt
validate_file=$input_dir/src-val.txt,$input_dir/tgt-val.txt
model_name_or_path=Salesforce/codet5-base
tokenizer_name=Salesforce/codet5-base
cache_path=$output_dir/cache_data
data_dir=$input_dir
pl=java

mkdir -p $output_dir

python ./run_apr.py \
--do_train \
--do_eval \
--model_type codet5 \
--model_name_or_path $model_name_or_path \
--tokenizer_name $tokenizer_name \
--train_filename $train_file \
--dev_filename $validate_file \
--output_dir $output_dir \
--max_source_length $max_source_length \
--max_target_length $max_target_length \
--beam_size $beam_size \
--train_batch_size $batch_size \
--eval_batch_size $batch_size \
--learning_rate $lr \
--num_train_epochs $epoch \
--summary_dir $summary_dir \
--cache_path $cache_path \
--data_dir $data_dir \
--res_dir $res_dir \
--task refine \
--lang $pl \
2>&1| tee $output_dir/$log_file