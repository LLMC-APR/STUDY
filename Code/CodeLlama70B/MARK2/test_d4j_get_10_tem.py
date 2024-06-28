import codecs
import os
import re
import sys
import json
import torch
import logging
import argparse
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, prepare_model_for_int8_training, set_peft_model_state_dict, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

logging.basicConfig(format = '%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt = '%m/%d/%Y %H:%M:%S',
                    level = logging.INFO)
logger = logging.getLogger(__name__)

# os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2,4,5,6,7'

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gradient_checkpointing", action="store_false", default=False)
    parser.add_argument("--lora_r", type=int, default=16)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    
    parser.add_argument("--base_model_name_or_path", type=str, default="codellama/CodeLlama-13b-hf")
    parser.add_argument("--peft_model_path", type=str, default="/")
    parser.add_argument("--push_to_hub", action="store_true", default=True)
    
    parser.add_argument("--model_name_or_path", type=str, default="codellama/CodeLlama-13b-hf")
    parser.add_argument("--test_filename", type=str, default="TRANSFER")
    parser.add_argument("--output_dir", type=str, default="TRANSFER")
    parser.add_argument("--max_source_length", type=int, default=512)
    parser.add_argument("--max_target_length", type=int, default=8)
    parser.add_argument("--beam_size", type=int, default=1)
    parser.add_argument("--output_size", type=int, default=1)
    parser.add_argument("--test_batch_size", type=int, default=1)
    
    return parser.parse_args()

def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )

def check_len(src_line, tokenizer, args, i):
    input_tokens = tokenizer.tokenize(src_line)
    
    if len(input_tokens) < args.max_source_length-128:
        return src_line
    else:
        # print('--------------------------------------')
        while 1:
            input_tokens = tokenizer.tokenize(src_line)

            if len(input_tokens) > args.max_source_length-128:
                bug_loc = input_tokens.index('start')
                # print(bug_loc, len(input_tokens))

                if bug_loc+256 < args.max_source_length-128:
                    src_line = src_line[:len(src_line)-2]
                elif bug_loc-256 > args.max_source_length-128:
                    src_line = src_line[1:]
                else:
                    src_line = src_line[1:]
                
            else:
                break
        
        # logger.info('%s, %s', str(i), str(len(input_tokens)))
        # print(len(input_tokens))

        return src_line


def model_inference():
    logger.info("-----------------------------------")
    logger.info("    Load Tokenizer and Model...    ")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, use_auth_token=True)
    
    nf4_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_use_double_quant=True,
    bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        quantization_config=nf4_config,
        # use_cache=not args.no_gradient_checkpointing,
        # load_in_4bit=True,
        device_map='auto'
    )    
    
    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    )

    model = get_peft_model(model, lora_config)

    # print_trainable_parameters(model)
    
    logger.info("-----------------------------------")
    logger.info("            Start Test...          ")
    
    assert len(args.test_filename.split(','))==2
    src_filename = args.test_filename.split(',')[0]
    trg_filename = args.test_filename.split(',')[1]
    
    
    with open(trg_filename, 'r') as data_file:
        data_lines = data_file.readlines()
    data_size = len(data_lines)
    
    pre_list = []
    tgt_list = []
    good = 0
    i = 0


    with open(src_filename, 'r') as f1,open(trg_filename, 'r') as f2:
        # all_data = f2.readlines()
        # data_size = len(all_data)
        # print('Dataset Size:')
        # print(len(f1), len(f2))
        logger.info("Data Size: %s", str(data_size)) 
        
        for line1,line2 in zip(f1,f2):
            i += 1

            # if i > 300 and i <= 560:
            #     pass
            # else:
            #     continue
                           
            
            src_dic = json.loads(line1)
            tgt_dic = json.loads(line2)
            

            src_line = src_dic['INPUT']
            
            tgt_line = tgt_dic['INPUT']
            
            tgt_list.append(str([tgt_line])+'\n')

            try:
                src_line = check_len(src_line, tokenizer, args, i)
            except:
                src_line = src_line[:args.max_source_length]
            
            B_INST, E_INST = "[INST]", "[/INST]"
            B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"
            
            
            # input_text = '\n<commit_before>\n' + src_line + '\n<commit_msg>' + template +  '\n<commit_after>\n'
            input_text =  tokenizer.bos_token + B_INST +'\n[bug_function]\n' + src_line + '\n[fix_code]\n' + E_INST
            # input_text =  tokenizer.bos_token + '\n<bug_function>\n' + src_line + '\n<repair_template>' + template + '\n<fix_code>\n'
            

            output_text = tgt_line
            
            # logger.info("Input Text: %s", str(input_text))   

            input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(0)
            pre_out = []
            
            if input_ids.size(1) >= args.max_source_length:
                pre_out.append('Input Too Long')
                logger.info(" %d/%d, Input_len= %s, Pre_Output = %s", i, data_size, str(input_ids.size(1)), str(['Input Too Long']))
                pre_list.append(str(pre_out) + '\n')
            else:
                try:
                    eos_id = tokenizer.convert_tokens_to_ids(tokenizer.eos_token)
                    tem_list = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
                    for tem_size in tem_list:
                        generated_ids = model.generate(
                            input_ids=input_ids, 
                            max_new_tokens=args.max_target_length, 
                            # num_beams=args.beam_size, 
                            num_return_sequences=args.output_size, 
                            # early_stopping=True,
                            temperature=tem_size,
                            top_p=1.0,
                            do_sample=True,
                            pad_token_id=eos_id, 
                            eos_token_id=eos_id
                        )
                        
                        # pre_out = []
                        for generated_id in generated_ids:
                            generated_text = tokenizer.decode(generated_id, skip_special_tokens=False)
                            text = generated_text.split(E_INST)[1]
                            text = text.replace(tokenizer.eos_token,'')
                            pre_out.append(text)
                except Exception as e:
                    pre_out.append('// Error\n')   
                    logger.info('%s', str(e))
                
                EM_result = 'bad'
                for pre_one in pre_out:
                    pre_txt = pre_one.replace('// fix_start\n','').replace('// fix_end\n', '').replace('\n','')
                    pre_txt = re.sub(r'\s+', '', pre_txt)

                    output_txt = output_text.replace('// fix_start\n','').replace('// fix_end\n', '').replace('\n','')
                    output_txt = re.sub(r'\s+', '', output_txt)

                    if pre_txt.strip() == output_txt.strip():
                        good += 1
                        EM_result = 'good'
                        break
                    
                logger.info(" %d/%d, Input_len=%s, EM_result=%s, Sample_size=%s, Pre_patch=%s", i, data_size, str(input_ids.size(1)), EM_result, str(len(pre_out)), str(pre_out))    
                pre_list.append(str(pre_out) + '\n')
    
            # save patch file
            if i % 4 == 0:
                with open(args.output_dir+'/pre_file.txt', 'w') as pre_file, open(args.output_dir+'/tgt_file.txt', 'w') as tgt_file:
                    for pre, tgt in zip(pre_list, tgt_list):
                        pre_file.write(pre)
                        tgt_file.write(tgt)

    logger.info(" Good = %d", good)
    logger.info(" All = %d", i)
    logger.info(" Repair Accuracy = %s", str(good/i))
    # logger.info(" Good = %d , Repair Accuracy = %s", good, str(good/data_size))           

    



if __name__ == '__main__':
    args = get_args()
    logger.info(args)
    
    # Setup CUDA, GPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # args.n_gpu = torch.cuda.device_count()
    # args.device = device
    # logger.info("device: %s, n_gpu: %s",device, args.n_gpu)
    
    model_dir = args.model_name_or_path
    input_file = args.test_filename
    output_dir = args.output_dir
        
    model_inference()
    

    
