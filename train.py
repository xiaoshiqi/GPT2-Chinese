# -*- encoding: utf-8 -*-
import transformers
import torch
import os
import json
import random
import copy
import numpy as np
import argparse
from datetime import datetime
from tqdm import tqdm
from torch.nn import DataParallel


def build_files(data_path, tokenized_data_path, full_tokenizer, n_ctx):
    pad = full_tokenizer.convert_tokens_to_ids('[PAD]')
    
    with open(data_path, 'r', encoding='utf8') as f:
        print('reading lines')
        lines = json.load(f)
        lines = [line.replace('\n', ' [SEP] ') for line in lines]

    if not os.path.exists(tokenized_data_path):
        os.mkdir(tokenized_data_path)

    lines = [full_tokenizer.tokenize(line) for line in lines ]
    lines = [full_tokenizer.convert_tokens_to_ids(line) for line in lines]
    
    full_line = []
    
    
    for line in lines:
        full_line.append(full_tokenizer.convert_tokens_to_ids('[MASK]'))  # token that represent the beginning of a couplet
        full_line.extend(line)
        
        temp = n_ctx-len(line)-2
        for _ in range(temp):
            full_line.append(pad)
        
        full_line.append(full_tokenizer.convert_tokens_to_ids('[CLS]'))  # token that represent the end of a couplet
    with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(0), 'w') as f:
        for id in full_line:
            f.write(str(id) + ' ')
                
    print('finish')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='0,1,2,3', type=str, required=False, help='cuda visible devices')
    parser.add_argument('--model_config', default='config/model_config.json', type=str, required=False,
                        help='path of the model configration file')
    parser.add_argument('--tokenizer_path', default='data/vocabs.txt', type=str, required=False, help='path of the vocabulary file')
    parser.add_argument('--raw_data_path', default='data/samples.json', type=str, required=False, help='path of the samples file')
    parser.add_argument('--tokenized_data_path', default='data/tokenized/', type=str, required=False,
                        help='save the tokenized samples file to this dir')
    parser.add_argument('--raw', action='store_true', help='do tokenize before training, no need if already tokenized with same configration')
    parser.add_argument('--epochs', default=24, type=int, required=False)
    parser.add_argument('--batch_size', default=16, type=int, required=False)
    parser.add_argument('--lr', default=2e-4, type=float, required=False)
    parser.add_argument('--warmup_steps', default=4000, type=int, required=False)
    parser.add_argument('--log_step', default=4000, type=int, required=False, help='period of reporting loss')
    parser.add_argument('--gradient_accumulation', default=1, type=int, required=False)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, required=False)
    parser.add_argument('--output_dir', default='model/', type=str, required=False, help='save the model to this dir')
    parser.add_argument('--pretrained_model', default='', type=str, required=False, help='pre-trained model dir')

    args = parser.parse_args()
    print('args:\n' + args.__repr__())

    from tokenizations import tokenization_bert

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    
    model_config = transformers.modeling_gpt2.GPT2Config.from_json_file(args.model_config)
    print('config:\n' + model_config.to_json_string())

    n_ctx = model_config.n_ctx
    full_tokenizer = tokenization_bert.BertTokenizer(vocab_file=args.tokenizer_path)
    full_tokenizer.max_len = 999999
    
    if torch.cuda.is_available():
        device = 'cuda' 
        print(torch.cuda.get_device_name(0))
    else:
        device = 'cpu'
        print(device)

    raw_data_path = args.raw_data_path
    tokenized_data_path = args.tokenized_data_path
    raw = args.raw
    epochs = args.epochs
    batch_size = args.batch_size
    lr = args.lr
    warmup_steps = args.warmup_steps
    log_step = args.log_step
    gradient_accumulation = args.gradient_accumulation
    max_grad_norm = args.max_grad_norm
    output_dir = args.output_dir
    assert log_step % gradient_accumulation == 0

    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    if raw:
        print('building files')
        build_files(data_path=raw_data_path, tokenized_data_path=tokenized_data_path, full_tokenizer=full_tokenizer, n_ctx=n_ctx)
        print('files built')

    if not args.pretrained_model:
        model = transformers.modeling_gpt2.GPT2LMHeadModel(config=model_config)
    else:
        model = transformers.modeling_gpt2.GPT2LMHeadModel.from_pretrained(args.pretrained_model)
    model.train()
    model.to(device)

    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    print('number of parameters: {}'.format(num_parameters))

    multi_gpu = False
    full_len = 0
    print('calculating total steps')
    
    with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(0), 'r') as f:
        full_len += len([int(item) for item in f.read().strip().split()])
            
    total_steps = int(full_len / n_ctx * epochs / batch_size / gradient_accumulation)
    print('total steps = {}'.format(total_steps))
    
    optimizer = transformers.AdamW(model.parameters(), lr=lr, correct_bias=True)
    scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=warmup_steps, t_total=total_steps)

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        
        device_ids = []
        for i in args.device.split(','):
            try:
                print(torch.cuda.get_device_name(int(i)))
                device_ids.append(int(i))
            except:
                pass
        model = DataParallel(model, device_ids=device_ids)
        multi_gpu = True
    print('starting training')
    overall_step = 0
    running_loss = 0
    
    
    with open(tokenized_data_path + 'tokenized_train_{}.txt'.format(0), 'r') as f:
        line = f.read().strip()
    tokens = line.split()
    tokens = [int(token) for token in tokens]
    start_point = 0
    samples = []
    
    while start_point < len(tokens) - n_ctx:
        samples.append(tokens[start_point: start_point + n_ctx])
        start_point += n_ctx   
    if start_point < len(tokens):
        samples.append(tokens[len(tokens)-n_ctx:])
        
    
    for epoch in range(epochs):
        print('epoch {}'.format(epoch + 1))
        now = datetime.now()
        print('time: {}'.format(now))
        
        samples2 = copy.deepcopy(samples)
        random.shuffle(samples2)
        
        for step in range(len(samples2) // batch_size):  # drop last
            #  prepare data
            batch = samples2[step * batch_size: (step + 1) * batch_size]
            batch_inputs = torch.tensor(batch).long().to(device)
            #  forward pass
            outputs = model.forward(input_ids=batch_inputs, labels=batch_inputs)
            loss, logits = outputs[:2]

            if multi_gpu:
                loss = loss.mean()
            if gradient_accumulation > 1:
                loss = loss / gradient_accumulation

            #  loss backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            #  optimizer step
            if (overall_step + 1) % gradient_accumulation == 0:
                running_loss += loss.item()
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()
            if (overall_step + 1) % log_step == 0:
                tb_writer.add_scalar('loss', loss.item() * gradient_accumulation, overall_step)
                print('now time: {}:{}. Step {} of epoch {}, loss {}'.format(
                    datetime.now().hour,
                    datetime.now().minute,
                    step + 1,
                    epoch + 1,
                    running_loss * gradient_accumulation / (log_step / gradient_accumulation)))
                running_loss = 0
            overall_step += 1

            
        print('saving model for epoch {}'.format(epoch + 1))
        temp_epoch = (epoch+1) % 2      # save disk space
        
        if not os.path.exists(output_dir + 'model_epoch{}'.format(temp_epoch)):
            os.mkdir(output_dir + 'model_epoch{}'.format(temp_epoch))
        model_to_save = model.module if hasattr(model, 'module') else model
        model_to_save.save_pretrained(output_dir + 'model_epoch{}'.format(temp_epoch))
        #torch.save(scheduler, output_dir + 'model_epoch{}/scheduler.pt'.format(temp_epoch))
        #torch.save(optimizer, output_dir + 'model_epoch{}/optimizer.pt'.format(temp_epoch))
        print('epoch {} finished'.format(epoch + 1))

        then = datetime.now()
        print('time: {}'.format(then))
        print('time for one epoch: {}'.format(then - now))

    print('training finished')


if __name__ == '__main__':
    main()
