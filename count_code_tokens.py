import json
import argparse
from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="llma")
    parser.add_argument("--input_data_fn", type=str, default="dev.top10.gpt.jsonl")
    parser.add_argument("--time", type=float, default=0.1)
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(llama_path):
    tokenizer = AutoTokenizer.from_pretrained(llama_path, mirror='tuna', cache_dir=llama_path)
    model = AutoModelForCausalLM.from_pretrained(llama_path, mirror='tuna', cache_dir=llama_path)
    # tokenizer = LlamaTokenizer.from_pretrained(llama_path)
    # model = LlamaForCausalLM.from_pretrained(llama_path)
    #tokenizer = LlamaTokenizer.from_pretrained("huggyllama/llama-7b", mirror='tuna', cache_dir=llama_path)
    #model = LlamaForCausalLM.from_pretrained("huggyllama/llama-7b", mirror='tuna', cache_dir=llama_path)
    model.half()
    model.cuda()
    return tokenizer, model


def load_data(input_fn):
    s_list = []
    with open(input_fn) as fin:
        for line in fin:
            s = json.loads(line)
            reslut = s['completion']
            s_list.append(reslut)
    return s_list


def main():
    args = get_args()
    llama_path = args.model_path
    tokenizer, model = get_tokenizer_and_model(llama_path)
    data_path_list = ["results/output_base_pass1.jsonl",
                      "results/output_base_pass5.jsonl",
                      "results/output_base_pass10.jsonl",
                      "results/output_fastcodellama_pass1.jsonl",
                      "results/output_fastcodellama_pass5.jsonl",
                      "results/output_fastcodellama_pass10.jsonl",
                      ]
    time_list = [2690.52, 13234.82, 35212.67, 1514.74, 8179.09, 13884.78]

    for i in range(6):
        data_path = data_path_list[i]
        time = time_list[i]

        s_list = load_data(data_path)
        alltokens = 0.0
        for s in s_list:
            gtokens = len(tokenizer.tokenize(s))
            alltokens += gtokens
        print('all tokens: ', alltokens)
        print('tokens/sec: ', alltokens/float(time))


if __name__ == '__main__':
    main()
