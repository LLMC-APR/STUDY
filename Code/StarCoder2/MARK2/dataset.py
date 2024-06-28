import torch
import codecs
import random
import json
from transformers import AutoTokenizer


class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_path, tokenizer, max_length, shuffle=False, load_range=None):
        self.data = []
        self.max_length = max_length
        
        assert len(file_path.split(','))==2
        src_filename = file_path.split(',')[0]
        trg_filename = file_path.split(',')[1]

        with open(src_filename, 'r') as f1,open(trg_filename, 'r') as f2:
            # print('Dataset Size:')
            # print(len(f1), len(f2))
            for line1,line2 in zip(f1,f2):
                src_dic = json.loads(line1)
                tgt_dic = json.loads(line2)
            
                src_line = src_dic['INPUT']
                tgt_line = tgt_dic['INPUT']
                
                inputs = '<commit_before>\n' + src_line + '\n<commit_after>\n' + tgt_line + tokenizer.eos_token
                outputs = tgt_line + tokenizer.eos_token

                

                inputs = tokenizer.encode(inputs, return_tensors='pt')
                outputs = tokenizer.encode(outputs, return_tensors='pt')
                if inputs.size(1) > max_length:
                    continue

                self.data.append({
                    'input_ids': inputs,
                    'labels': torch.cat([torch.zeros(1, inputs.size(1) - outputs.size(1)).fill_(-100).long(), outputs], dim=1),
                    'attention_mask': torch.ones(inputs.size()).long()
                })

                # if len(self.data) % 10000 == 0:
                #     print('finish loading:', len(self.data))
                
                # if load_range is not None and len(self.data) == load_range[1]:
                #     break
        
        if shuffle:
            random.seed(7)
            random.shuffle(self.data)

        print(file_path, 'total size:', len(self.data))
        if load_range is not None:
            self.data = self.data[load_range[0]: ]
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, item):
        return self.data[item]


def custom_collate(batch):
    batch_data = {'input_ids': [], 'labels': [], 'attention_mask': []}
    max_len = max([b['input_ids'].size(1) for b in batch])
    for b in batch:
        batch_data['input_ids'].append(torch.cat([b['input_ids'], torch.zeros(1, max_len - b['input_ids'].size(1)).fill_(2).long()], dim=1))
        batch_data['labels'].append(torch.cat([b['labels'], torch.zeros(1, max_len - b['labels'].size(1)).fill_(-100).long()], dim=1))
        batch_data['attention_mask'].append(torch.cat([b['attention_mask'], torch.zeros(1, max_len - b['attention_mask'].size(1))], dim=1))
    batch_data['input_ids'] = torch.cat(batch_data['input_ids'], dim=0)
    batch_data['labels'] = torch.cat(batch_data['labels'], dim=0)
    batch_data['attention_mask'] = torch.cat(batch_data['attention_mask'], dim=0)
    return batch_data
