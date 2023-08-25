from transformers import LlamaForCausalLM, LlamaTokenizer
import torch
import json
from collections import defaultdict
from tqdm import tqdm

import time
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="llma")
    parser.add_argument("--input_data_fn", type=str, default="dev.top10.gpt.jsonl")
    parser.add_argument("--time", type=float, default=0.1)
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(llama_path):
   # tokenizer = LlamaTokenizer.from_pretrained(llama_path)
   # model = LlamaForCausalLM.from_pretrained(llama_path)
    tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", mirror='tuna', cache_dir=llama_path)
    model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", mirror='tuna', cache_dir=llama_path)
    model.half()
    model.cuda()
    return tokenizer, model


def load_data(input_fn, tokenizer):
    s_list = []
    with open(input_fn) as fin:
        for line in fin:
            s = json.loads(line)
            reslut = s['completion']
            s_list.append(reslut)
    return s_list


def main():
    args = get_args()
    time = args.time
    llama_path = args.model_path
    tokenizer, model = get_tokenizer_and_model(llama_path)
    input_fn = args.input_data_fn


    while True:
        data_path = input('data_path: ')
        if data_path=="break":
            print('end')
            break
        time = input('time: ')

        s_list = load_data(data_path, tokenizer)
        alltokens = 0.0
        for s in s_list:
            gtokens = len(tokenizer.tokenize(s))
            alltokens += gtokens
        print('all tokens: ', alltokens)
        print('tokens/sec: ', alltokens/float(time))

if __name__ == '__main__':
    main()