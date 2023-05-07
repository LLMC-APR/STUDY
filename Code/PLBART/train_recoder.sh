
# data_dir=/SS970evo/datasets/Tufano_dataset/datasets/preprocess/50-100/mark2_src
# output_dir=/SS970evo/datasets/Tufano_dataset/datasets/result_PLBART/50-100/mark2_src/model_set1
data_dir=/SS970evo/datasets/Recoder_dataset/mark2
output_dir=/SS970evo/datasets/Recoder_dataset/result_PLBART/mark2/model_CSNET

max_source_length=512
max_target_length=256
beam_size=5
train_batch_size=16 # 8
eval_batch_size=16
learning_rate=5e-5
num_train_epochs=30

mkdir -p $output_dir

python run_sequencer.py \
        --do_train \
        --do_eval \
        --model_type bart \
        --model_name_or_path ../../../PLBART/pretrain/checkpoint_356_100000.pt \
        --tokenizer_name ../../../PLBART/sentencepiece/sentencepiece.bpe.model \
        --train_filename $data_dir/src-train.txt,$data_dir/tgt-train.txt \
        --dev_filename $data_dir/src-val.txt,$data_dir/tgt-val.txt \
        --output_dir $output_dir \
        --max_source_length $max_source_length \
        --max_target_length $max_target_length \
        --beam_size $beam_size \
        --train_batch_size $train_batch_size \
        --eval_batch_size $eval_batch_size \
        --learning_rate $learning_rate \
        --num_train_epochs $num_train_epochs \
        2>&1 | tee $output_dir/train.log