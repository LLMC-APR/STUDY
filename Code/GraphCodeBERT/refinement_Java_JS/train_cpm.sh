lr=5e-5  
batch_size=16
beam_size=5
source_length=512
target_length=256
output_dir=/dataset/CPatMiner_dataset/result_GraphCodeBERT/mark2/model_set1
input_dir=/dataset/CPatMiner_dataset/mark2
train_file=$input_dir/src-train.txt,$input_dir/tgt-train.txt
dev_file=$input_dir/src-val.txt,$input_dir/tgt-val.txt
log_file=train.log
epochs=30 
pretrained_model=microsoft/graphcodebert-base

mkdir -p $output_dir
python run_recoder.py --do_train --do_eval --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --train_filename $train_file --dev_filename $dev_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --train_batch_size $batch_size --eval_batch_size $batch_size --learning_rate $lr --num_train_epochs $epochs 2>&1| tee $output_dir/$log_file