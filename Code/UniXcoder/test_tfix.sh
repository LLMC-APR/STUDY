
lr=5e-5
batch_size=32 #64
beam_size=5
source_length=512
target_length=128
output_dir=/dataset/TFix_dataset/result_UniXcoder/mark2_src/model_set2
input_dir=/dataset/TFix_dataset/mark2
dev_file=None
test_file=$input_dir/src-test.txt,$input_dir/tgt-test.txt
checkpoint_type=best-bleu # best-ppl/bleu last
load_model_path=$output_dir/checkpoint-$checkpoint_type/pytorch_model.bin # checkpoint-best-bleu
log_file=test_$checkpoint_type.log
pretrained_model=microsoft/unixcoder-base


python run_demo.py --do_test --model_name_or_path $pretrained_model --load_model_path $load_model_path --dev_filename $dev_file --test_filename $test_file --output_dir $output_dir --max_source_length $source_length --max_target_length $target_length --beam_size $beam_size --eval_batch_size $batch_size 2>&1| tee $output_dir/$log_file