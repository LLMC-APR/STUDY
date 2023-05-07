
lr=5e-5
batch_size=1
beam_size=100
source_length=512
target_length=512
model_dir=/dataset/Recoder_dataset/result_GraphCodeBERT/mark2/model_set1_try
input_dir=/dataset/Defects4J_dataset/mark2_D4j_v2
output_dir=$model_dir/D4J_V2_bleu
# dev_file=data/$scale/valid.buggy-fixed.buggy,data/$scale/valid.buggy-fixed.fixed
dev_file=None
test_file=$input_dir/src-test.txt,$input_dir/tgt-test.txt
checkpoint_type=best-bleu  # best-ppl/bleu last
load_model_path=$model_dir/checkpoint-$checkpoint_type/pytorch_model.bin # checkpoint for test
log_file=test_$checkpoint_type.log
pretrained_model=microsoft/graphcodebert-base

python run_sequencer.py --do_test --model_type roberta --model_name_or_path $pretrained_model --tokenizer_name microsoft/graphcodebert-base --config_name microsoft/graphcodebert-base --load_model_path $load_model_path --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 2>&1| tee $output_dir/$log_file