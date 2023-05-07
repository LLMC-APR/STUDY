# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for language modeling on a text file (GPT, GPT-2, BERT, RoBERTa).
GPT and GPT-2 are fine-tuned using a causal language modeling (CLM) loss while BERT and RoBERTa are fine-tuned
using a masked language modeling (MLM) loss.
"""

import os
import logging
import argparse
import math
import numpy as np
from tqdm import tqdm
import multiprocessing
import time

import torch
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import AdamW, get_linear_schedule_with_warmup
from models import build_or_load_gen_model
from evaluator import smooth_bleu
from evaluator.CodeBLEU import calc_code_bleu
from evaluator.bleu import _bleu
from utils import get_filenames, get_elapse_time, load_and_cache_gen_data
from configs import add_args, set_seed, set_dist

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)


os.environ["CUDA_VISIBLE_DEVICES"] = '0'

def eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer):
    eval_sampler = SequentialSampler(eval_data)
    eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                 num_workers=4, pin_memory=True)
    # Start evaluating model
    logger.info("  " + "***** Running ppl evaluation *****")
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    eval_loss, batch_num = 0, 0
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval ppl"):
        batch = tuple(t.to(args.device) for t in batch)
        source_ids, target_ids = batch
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        target_mask = target_ids.ne(tokenizer.pad_token_id)

        with torch.no_grad():
            if args.model_type == 'roberta':
                loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                   target_ids=target_ids, target_mask=target_mask)
            else:
                outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                labels=target_ids, decoder_attention_mask=target_mask)
                loss = outputs.loss
        # eval_loss += loss.item()
        eval_loss += torch.mean(loss).item()
        batch_num += 1
    eval_loss = eval_loss / batch_num
    eval_ppl = round(np.exp(eval_loss), 5)
    return eval_ppl


def eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, split_tag, criteria):
    logger.info("  ***** Running bleu evaluation on {} data*****".format(split_tag))
    logger.info("  Num examples = %d", len(eval_examples))
    logger.info("  Batch size = %d", args.eval_batch_size)
    eval_sampler = SequentialSampler(eval_data)
    if args.data_num == -1:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size,
                                     num_workers=4, pin_memory=True)
    else:
        eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=args.eval_batch_size)
    
    
    change = False
    if isinstance(model,torch.nn.DataParallel):
        model = model.module
        change = True
        
    model.eval()
    pred_ids = []
    pred_all = []
    bleu, codebleu = 0.0, 0.0
    
    # for batch in tqdm(eval_dataloader, total=len(eval_dataloader), desc="Eval bleu for {} set".format(split_tag)):
    for batch in tqdm(eval_dataloader, total=len(eval_dataloader)):
        source_ids = batch[0].to(args.device)
        source_mask = source_ids.ne(tokenizer.pad_token_id)
        with torch.no_grad():
            if args.model_type == 'roberta':
                preds = model(source_ids=source_ids, source_mask=source_mask)

                top_preds = [pred[0].cpu().numpy() for pred in preds]
            else:
                
                preds = model.generate(source_ids,
                                        attention_mask=source_mask,
                                        use_cache=True,
                                        do_sample=False,
                                        num_beams=args.beam_size,
                                        num_return_sequences=args.beam_size,
                                        # num_return_sequences=10,
                                        early_stopping=args.task == 'refine',
                                        max_length=args.max_target_length)
                # print(preds)
                top_preds = list(preds.cpu().numpy())
                texts = [tokenizer.decode(t, skip_special_tokens=True, clean_up_tokenization_spaces=False) for t in top_preds]
                # print(texts)
                # logger.info("%s", str(texts[:50]))
                # logger.info("%s", str(texts[50:]))
                # pred_all.append(texts[:100])
                # pred_all.append(texts[100:])
                pred_all.append(texts)
            
    
    if change:
        model = torch.nn.DataParallel(model)

    output_fn = os.path.join(args.res_dir, "test_{}.output".format(criteria))
    gold_fn = os.path.join(args.res_dir, "test_{}.gold".format(criteria))
    # src_fn = os.path.join(args.res_dir, "test_{}.src".format(criteria))

    if args.task in ['defect']:
        target_dict = {0: 'false', 1: 'true'}
        # golds = [target_dict[ex.target] for ex in eval_examples]
        # eval_acc = np.mean([int(p == g) for p, g in zip(pred_nls, golds)])
        # result = {'em': eval_acc * 100, 'bleu': 0, 'codebleu': 0}

        # with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1, open(src_fn, 'w') as f2:
        #     for pred_nl, gold in zip(pred_nls, eval_examples):
        #         f.write(pred_nl.strip() + '\n')
        #         f1.write(target_dict[gold.target] + '\n')
        #         f2.write(gold.source.strip() + '\n')
        #     logger.info("Save the predictions into %s", output_fn)
    else:
        dev_accs, predictions = [], []
        good, bad = 0, 0
        with open(output_fn, 'w') as f, open(gold_fn, 'w') as f1:
            for pred_nl, gold in zip(pred_all, eval_examples):
                for one_pred_nl in pred_nl:
                    if gold.target.strip() == one_pred_nl.strip():
                        good += 1
                        break
                    
                f.write(str(pred_nl) + '\n')
                f1.write(gold.target.strip() + '\n')
                

    bad = len(eval_examples) - good   
    result = good / (good+bad)
    logger.info("***** Eval results *****")
    logger.info("  good fixes: %s", str(good))
    logger.info("  repait accuracy: %s", str(result))

    return result


def main():
    parser = argparse.ArgumentParser()
    
    
    parser.add_argument("--task", type=str, required=True, default='refine',
                        choices=['summarize', 'concode', 'translate', 'refine', 'defect', 'clone', 'multi_task'])
    parser.add_argument("--sub_task", type=str, default='')
    parser.add_argument("--lang", type=str, default='')
    parser.add_argument("--eval_task", type=str, default='')
    parser.add_argument("--model_type", default="codet5", type=str, choices=['roberta', 'bart', 'codet5'])
    parser.add_argument("--add_lang_ids", action='store_true')
    parser.add_argument("--data_num", default=-1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--num_train_epochs", default=100, type=int)
    parser.add_argument("--patience", default=5, type=int)
    parser.add_argument("--cache_path", type=str, required=True)
    parser.add_argument("--summary_dir", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--res_dir", type=str, required=True)
    parser.add_argument("--res_fn", type=str, default='')
    parser.add_argument("--add_task_prefix", action='store_true', help="Whether to add task prefix for t5 and codet5")
    parser.add_argument("--save_last_checkpoints", default='save', action='store_true')
    parser.add_argument("--always_save_model", default='save', action='store_true')
    parser.add_argument("--do_eval_bleu", default='save', action='store_true', help="Whether to evaluate bleu on dev set.")

    ## Required parameters
    parser.add_argument("--model_name_or_path", default="roberta-base", type=str,
                        help="Path to pre-trained model: e.g. roberta-base")
    parser.add_argument("--output_dir", default=None, type=str, required=True,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--load_model_path", default=None, type=str,
                        help="Path to trained model: Should contain the .bin files")
    ## Other parameters
    parser.add_argument("--train_filename", default=None, type=str,
                        help="The train filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--dev_filename", default=None, type=str,
                        help="The dev filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--test_filename", default=None, type=str,
                        help="The test filename. Should contain the .jsonl files for this task.")
    parser.add_argument("--checkpoint_model_type", default=None, type=str,
                        help="The checkpoint_model_type in testing, file_type: best-bleu/ppl/last.")
    
    
    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")
    parser.add_argument("--tokenizer_name", default="roberta-base", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")
    parser.add_argument("--max_source_length", default=64, type=int,
                        help="The maximum total source sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")
    parser.add_argument("--max_target_length", default=32, type=int,
                        help="The maximum total target sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--do_train", action='store_true',
                        help="Whether to run eval on the train set.")
    parser.add_argument("--do_eval", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_test", action='store_true',
                        help="Whether to run eval on the dev set.")
    parser.add_argument("--do_lower_case", action='store_true',
                        help="Set this flag if you are using an uncased model.")
    parser.add_argument("--no_cuda", action='store_true',
                        help="Avoid using CUDA when available")

    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Batch size per GPU/CPU for evaluation.")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument("--learning_rate", default=5e-5, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--beam_size", default=10, type=int,
                        help="beam size for beam search")
    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")

    parser.add_argument("--save_steps", default=-1, type=int, )
    parser.add_argument("--log_steps", default=-1, type=int, )
    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform. Override num_train_epochs.")
    parser.add_argument("--eval_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--train_steps", default=-1, type=int,
                        help="")
    parser.add_argument("--warmup_steps", default=100, type=int,
                        help="Linear warmup over warmup_steps.")
    parser.add_argument("--local_rank", type=int, default=-1,
                        help="For distributed training: local_rank")
    parser.add_argument('--seed', type=int, default=1234,
                        help="random seed for initialization")
    
    # print arguments
    args = parser.parse_args()
    # parser = argparse.ArgumentParser()
    # args = add_args(parser)
    logger.info(args)
    
    t0 = time.time()
    
    # Setup CUDA, GPU & distributed training
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        args.n_gpu = torch.cuda.device_count()
    else:  # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend='nccl')
        args.n_gpu = 1
    logger.warning("Process rank: %s, device: %s, n_gpu: %s, distributed training: %s",
                   args.local_rank, device, args.n_gpu, bool(args.local_rank != -1))
    args.device = device

    # Set seed
    set_dist(args)
    set_seed(args)
    
    # make dir if output_dir not exist
    if os.path.exists(args.output_dir) is False:
        os.makedirs(args.output_dir)
    # make dir if cache_path not exist
    if os.path.exists(args.cache_path) is False:
        os.makedirs(args.cache_path)
    
    config, model, tokenizer = build_or_load_gen_model(args)
    # tokenizer.add_tokens(["<BUGS>", "<BUGE>", "<FIXS>", "<FIXE>"])
    model.to(args.device)
    if args.local_rank != -1:
        # Distributed training
        try:
            from apex.parallel import DistributedDataParallel as DDP
        except ImportError:
            raise ImportError(
                "Please install apex from https://www.github.com/nvidia/apex to use distributed and fp16 training.")

        model = DDP(model)
    elif args.n_gpu > 1:
        # for DataParallel
        model = torch.nn.DataParallel(model)
    
    
    pool = multiprocessing.Pool(args.cpu_cont)
    # args.train_filename, args.dev_filename, args.test_filename = get_filenames(args.data_dir, args.task, args.sub_task)
    # fa = open(os.path.join(args.output_dir, 'summary.log'), 'a+')

    if args.do_train:
        # if args.local_rank in [-1, 0] and args.data_num == -1:
        #     summary_fn = args.summary_dir
        #     tb_writer = SummaryWriter(summary_fn)

        # Prepare training data loader
        train_examples, train_data = load_and_cache_gen_data(args, args.train_filename, pool, tokenizer, 'train')
        if args.local_rank == -1:
            train_sampler = RandomSampler(train_data)
        else:
            train_sampler = DistributedSampler(train_data)
        # train_sampler = RandomSampler(train_data) if args.local_rank == -1 else DistributedSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.train_batch_size,
                                      num_workers=4, pin_memory=True)

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': args.weight_decay},
            {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
        num_train_optimization_steps = args.num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=args.warmup_steps,
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", train_example_num)
        logger.info("  Batch size = %d", args.train_batch_size)
        logger.info("  Batch num = %d", math.ceil(train_example_num / args.train_batch_size))
        logger.info("  Num epoch = %d", args.num_train_epochs)

        dev_dataset = {}
        global_step, best_bleu_em, best_ppl = 0, -1, 1e6
        not_loss_dec_cnt, not_bleu_em_inc_cnt = 0, 0 if args.do_eval_bleu else 1e6

        for cur_epoch in range(args.start_epoch, int(args.num_train_epochs)):
            logger.info("  Now epoch = %d", cur_epoch)
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            model.train()
            for step, batch in enumerate(bar):
                batch = tuple(t.to(args.device) for t in batch)
                source_ids, target_ids = batch
                source_mask = source_ids.ne(tokenizer.pad_token_id)
                target_mask = target_ids.ne(tokenizer.pad_token_id)

                if args.model_type == 'roberta':
                    loss, _, _ = model(source_ids=source_ids, source_mask=source_mask,
                                       target_ids=target_ids, target_mask=target_mask)
                else:
                    outputs = model(input_ids=source_ids, attention_mask=source_mask,
                                    labels=target_ids, decoder_attention_mask=target_mask)
                    loss = outputs.loss

                if args.n_gpu > 1:
                    loss = loss.mean()  # mean() to average on multi-gpu.
                if args.gradient_accumulation_steps > 1:
                    loss = loss / args.gradient_accumulation_steps
                tr_loss += loss.item()

                nb_tr_examples += source_ids.size(0)
                nb_tr_steps += 1
                loss.backward()

                if nb_tr_steps % args.gradient_accumulation_steps == 0:
                    # Update parameters
                    optimizer.step()
                    optimizer.zero_grad()
                    scheduler.step()
                    global_step += 1
                    train_loss = round(tr_loss * args.gradient_accumulation_steps / (nb_tr_steps + 1), 4)
                    bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))

            if args.do_eval and (int(cur_epoch) % 2 == 0 or int(cur_epoch) > int(args.num_train_epochs) -5 ): # every 2 epoch do eval, last 5 epoch all do eval
                # Eval model with dev dataset
                logger.info("  Start Eval...")
                if 'dev_loss' in dev_dataset:
                    eval_examples, eval_data = dev_dataset['dev_loss']
                else:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev')
                    dev_dataset['dev_loss'] = eval_examples, eval_data

                eval_ppl = eval_ppl_epoch(args, eval_data, eval_examples, model, tokenizer)
                result = {'epoch': cur_epoch, 'global_step': global_step, 'eval_ppl': eval_ppl}
                for key in sorted(result.keys()):
                    logger.info("  %s = %s", key, str(result[key]))
                logger.info("  " + "*" * 20)
                # if args.data_num == -1:
                #     tb_writer.add_scalar('dev_ppl', eval_ppl, cur_epoch)

                # save last checkpoint
                if args.save_last_checkpoints:
                    last_output_dir = os.path.join(args.output_dir, 'checkpoint-last')
                    if not os.path.exists(last_output_dir):
                        os.makedirs(last_output_dir)
                    model_to_save = model.module if hasattr(model, 'module') else model
                    output_model_file = os.path.join(last_output_dir, "pytorch_model.bin")
                    torch.save(model_to_save.state_dict(), output_model_file)
                    logger.info("Save the last model into %s", output_model_file)

                if eval_ppl < best_ppl:
                    not_loss_dec_cnt = 0
                    logger.info("  Best ppl:%s", eval_ppl)
                    logger.info("  " + "*" * 20)
                    # fa.write("[%d] Best ppl changed into %.4f\n" % (cur_epoch, eval_ppl))
                    best_ppl = eval_ppl

                    # Save best checkpoint for best ppl
                    output_dir = os.path.join(args.output_dir, 'checkpoint-best-ppl')
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)
                    if args.always_save_model:
                        model_to_save = model.module if hasattr(model, 'module') else model
                        output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                        torch.save(model_to_save.state_dict(), output_model_file)
                        logger.info("Save the best ppl model into %s", output_model_file)
                else:
                    not_loss_dec_cnt += 1
                    logger.info("ppl does not decrease for %d epochs", not_loss_dec_cnt)
                    if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                        early_stop_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                            cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                        logger.info(early_stop_str)
                        # fa.write(early_stop_str)
                        break
                logger.info("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
                
                if args.do_eval_bleu:
                    eval_examples, eval_data = load_and_cache_gen_data(args, args.dev_filename, pool, tokenizer, 'dev',
                                                                       only_src=True, is_sample=True)

                    result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'dev', 'eval')
                    dev_bleu, dev_em = result['bleu'], result['em']
                    if args.task in ['summarize']:
                        dev_bleu_em = dev_bleu
                    elif args.task in ['defect']:
                        dev_bleu_em = dev_em
                    else:
                        dev_bleu_em = dev_bleu + dev_em
                    # if args.data_num == -1:
                    #     tb_writer.add_scalar('dev_bleu_em', dev_bleu_em, cur_epoch)
                        # tb_writer.add_scalar('dev_em', dev_em, cur_epoch)
                    if dev_bleu_em > best_bleu_em:
                        not_bleu_em_inc_cnt = 0
                        logger.info("  [%d] Best bleu+em: %.2f (bleu: %.2f, em: %.2f)",
                                    cur_epoch, dev_bleu_em, dev_bleu, dev_em)
                        logger.info("  " + "*" * 20)
                        best_bleu_em = dev_bleu_em
                        # fa.write("[%d] Best bleu+em changed into %.2f (bleu: %.2f, em: %.2f)\n" % (
                        #     cur_epoch, best_bleu_em, dev_bleu, dev_em))
                        # Save best checkpoint for best bleu
                        output_dir = os.path.join(args.output_dir, 'checkpoint-best-bleu')
                        if not os.path.exists(output_dir):
                            os.makedirs(output_dir)
                        if args.data_num == -1 or args.always_save_model:
                            model_to_save = model.module if hasattr(model, 'module') else model
                            output_model_file = os.path.join(output_dir, "pytorch_model.bin")
                            torch.save(model_to_save.state_dict(), output_model_file)
                            logger.info("Save the best bleu model into %s", output_model_file)
                    else:
                        not_bleu_em_inc_cnt += 1
                        logger.info("Bleu does not increase for %d epochs", not_bleu_em_inc_cnt)
                        # fa.write(
                        #     "[%d] Best bleu+em (%.2f) does not drop changed for %d epochs, cur bleu+em: %.2f (bleu: %.2f, em: %.2f)\n" % (
                        #         cur_epoch, best_bleu_em, not_bleu_em_inc_cnt, dev_bleu_em, dev_bleu, dev_em))
                        if all([x > args.patience for x in [not_bleu_em_inc_cnt, not_loss_dec_cnt]]):
                            stop_early_str = "[%d] Early stop as not_bleu_em_inc_cnt=%d, and not_loss_dec_cnt=%d\n" % (
                                cur_epoch, not_bleu_em_inc_cnt, not_loss_dec_cnt)
                            logger.info(stop_early_str)
                            # fa.write(stop_early_str)
                            break
            logger.info("***** CUDA.empty_cache() *****")
            torch.cuda.empty_cache()

        # if args.local_rank in [-1, 0] and args.data_num == -1:
        #     tb_writer.close()
        logger.info("Finish training and take %s", get_elapse_time(t0))

    if args.do_test:
        logger.info("  " + "***** Testing *****")
        logger.info("  Batch size = %d", args.eval_batch_size)

        for criteria in ['best-bleu']:
            criteria = args.checkpoint_model_type  # best-bleu best-ppl best-last
            file = os.path.join(args.output_dir, 'checkpoint-{}/pytorch_model.bin'.format(criteria))
            logger.info("Reload model from {}".format(file))
            
            if hasattr(model, 'module'):
                model.module.load_state_dict(torch.load(file))
            else:
                model.load_state_dict(torch.load(file))
            eval_examples, eval_data = load_and_cache_gen_data(args, args.test_filename, pool, tokenizer, 'test',
                                                               only_src=True, is_sample=False)
            result = eval_bleu_epoch(args, eval_data, eval_examples, model, tokenizer, 'test', criteria)
            
            
            
            if args.res_fn:
                with open(args.res_fn, 'a+') as f:
                    f.write('[Time: {}] {}\n'.format(get_elapse_time(t0), file))
                    
    logger.info("Finish and take {}".format(get_elapse_time(t0)))
    # fa.write("Finish and take {}".format(get_elapse_time(t0)))
    # fa.close()
    logger.info("  " + "Test End" +"*" * 20)


if __name__ == "__main__":
    main()
