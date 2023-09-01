from transformers import LlamaForCausalLM, LlamaTokenizer, AutoTokenizer, AutoModelForCausalLM
import torch
import json
from collections import defaultdict
from tqdm import tqdm
import time
import argparse
import jsonlines

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--type", type=str, default="base")
    parser.add_argument("--n", type=int, default=1)
    parser.add_argument("--k", type=int, default=20)
    parser.add_argument("--append_docs", action="store_true")
    parser.add_argument("--input_data_fn", type=str, default="./data/HumanEval.jsonl")
    parser.add_argument("--output_data_fn", type=str, default="./result/samples.jsonl")
    parser.add_argument("--forced_decoding", action="store_true")
    parser.add_argument("--passk", type=int, default=1)
    args = parser.parse_args()
    return args


def get_tokenizer_and_model(llama_path):
    tokenizer = AutoTokenizer.from_pretrained(llama_path)
    model = AutoModelForCausalLM.from_pretrained(llama_path)
    model.half()
    model.cuda()
    return tokenizer, model


def truncate(doc, tokenizer, max_tokens=1024):
    if max_tokens <= 0:
        return doc
    tokens = tokenizer.tokenize(doc)[:max_tokens]
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    doc = tokenizer.decode(token_ids)
    return doc

def load_data_humaneval(input_fn, tokenizer):
    s_list = []
    with open(input_fn) as fin:
        for line in fin:
            s = json.loads(line)
            # 目前还没有’docs‘,后续会生成‘docs’,截断操作因为没有输入到context里，所以暂时不需要
            # for i, doc in enumerate(s['docs']):
            #     s['docs'][i] = truncate(doc, tokenizer, max_tokens=768)
            s_list.append(s)
    return s_list
    # return s_list

def temperature_sampling(logits, temperature=0.7):
    scaled_logits = logits / temperature
    probs = torch.softmax(scaled_logits, dim=-1)
    probs = probs.view(-1, probs.size(-1))
    sampled_token = torch.multinomial(probs, num_samples=1)
    sampled_token = sampled_token.view(1, -1)
    return sampled_token


def get_ngrams(tokens, n):
    ngram_list = []
    for i in range(len(tokens) - n + 1):
        ngram = ' '.join(tokens[i:i + n])
        ngram_list.append(ngram)
    return ngram_list


def match_length(seq_a, seq_b):
    l = 0
    for i in range(min(len(seq_a), len(seq_b))):
        if seq_a[i] != seq_b[i]:
            break
        l += 1
    return l


def prepare_ngrams(s, n, tokenizer, max_n=0):

    gtokens = None
    g_ngrams_list = None

    doc_list = []
    doc_token_id_list = []
    doc_ngrams_list = []

    return {"target_ngrams": g_ngrams_list, "doc_list": doc_list, "doc_ngrams": doc_ngrams_list,
            "target_tokens": gtokens, "doc_token_id_list": doc_token_id_list}


def base_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N=0, block_K=0, pass_k=1, forced_decoding=False,
                  ngrams_cache=None):
    generate_ids_list = []
    for idx in range(pass_k):
        prepend_ids = input_ids.cuda()
        generate_ids = None
        past_key_values = None
        if forced_decoding:
            eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
            gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)

        step = 0
        step_length = 1

        while True:
            with torch.no_grad():
                output = model(input_ids=prepend_ids,
                               past_key_values=past_key_values,
                               return_dict=True,
                               use_cache=True)
                logits = output['logits']
                logits = logits[:, -1:, :]
                # output_ids = temperature_sampling(logits, temperature=0.7)
                output_ids = torch.argmax(logits, dim=-1)

                if forced_decoding:
                    output_ids = gen_texts_ids[:, step:step + step_length].to(output_ids.device)
                prepend_ids = output_ids
                if generate_ids is None:
                    generate_ids = output_ids
                else:
                    generate_ids = torch.concat([generate_ids, output_ids], dim=1)
                past_key_values = output['past_key_values']
                output_ids = output_ids.cpu().numpy()
                step += 1
                if output_ids[0][-1] == tokenizer.eos_token_id or step > 512:  # llama遇到代码b
                    break
                # print(step, output_ids[0][-1])
        generate_ids_list.append(generate_ids.cpu())
    return generate_ids_list, 0


def match_prefix(g_ngrams_list, doc_ngrams_list, step):
    for g_ngrams, doc_ngrams in zip(g_ngrams_list, doc_ngrams_list):
        n = g_ngrams[0]
        g_ngrams = g_ngrams[1]
        doc_ngrams = doc_ngrams[1]
        if step < n:  # step > n 才可以
            continue
        if g_ngrams[step - n] in doc_ngrams.keys():
            return n, g_ngrams, doc_ngrams
    return 0, None, None


def make_kv_tupe(input_kv, step_length, accepted_step_length):
    kv_list = []
    for kv in input_kv:
        l = kv.shape[2]
        kv_list.append(kv[:, :, :l - step_length + accepted_step_length, :])
    return tuple(kv_list)


def make_past_key_values(past_key_values, step_length, accepted_step_length):
    if step_length == accepted_step_length:
        return past_key_values
    pkv_list = []
    for kv in past_key_values:
        kv = make_kv_tupe(kv, step_length, accepted_step_length)
        pkv_list.append(kv)
    return tuple(pkv_list)


def fastcodellama_generate_one(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, pass_k=1, forced_decoding=False,
                  ngrams_cache=None):
    generate_ids_list = []
    doc_list = []
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    accepted_tokens = 0

    prepend_ids = input_ids.cuda()
    generate_ids = None
    past_key_values = None

    if forced_decoding:
        gtokens = ngrams_cache["target_tokens"]
        g_ngrams_list = ngrams_cache["target_ngrams"]
        eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
        gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)
    else:
        gtokens = []
        g_ngrams_list = []
        for nlist in doc_ngrams_list:  # doc_ngrams_list.append([l, doc_ngrams])
            g_ngrams_list.append((nlist[0], []))

    step = 0
    while True:
        prefix_n, g_ngrams, doc_ngrams = match_prefix(g_ngrams_list, doc_ngrams_list, step)
        if prefix_n > 0:
            copy_mode = True
            trigger_ngram = g_ngrams[step - prefix_n]
            i, j = doc_ngrams[trigger_ngram][0]
            copied_ids = doc_token_id_list[i][j + prefix_n:j + prefix_n + block_K - 1]
            step_length = 1 + len(copied_ids)
            copied_ids = torch.tensor([copied_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
            prepend_ids = torch.concat([prepend_ids, copied_ids], dim=-1)
        else:
            step_length = 1
            copy_mode = False
        with torch.no_grad():
            output = model(input_ids=prepend_ids,
                           past_key_values=past_key_values,
                           return_dict=True,
                           use_cache=True)
            logits = output['logits'][:, -step_length:, :]
            output_ids = temperature_sampling(logits, temperature=0.7)
            accepted_step_length = step_length
            past_key_values = output['past_key_values']
            if copy_mode:
                iids = prepend_ids.cpu().numpy()[0]
                oids = output_ids.cpu().numpy()[0]
                real_output_ids = [oids[0]]
                for pos in range(1, len(oids)):
                    if oids[pos - 1] == iids[pos] and oids[pos - 1] != tokenizer.eos_token_id:
                        real_output_ids.append(oids[pos])
                    else:
                        break
                accepted_step_length = len(real_output_ids)
                past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)
                if accepted_step_length < step_length:
                    output_ids = output_ids[:, :accepted_step_length]
            step += accepted_step_length
            prepend_ids = output_ids[:, -1:]
            if generate_ids is None:
                generate_ids = output_ids
            else:
                generate_ids = torch.concat([generate_ids, output_ids], dim=1)
            output_ids = output_ids.cpu().numpy()
            output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
            if not forced_decoding:
                gtokens += output_tokens
                g_ngrams_list.clear()
                for pos in range(len(doc_ngrams_list)):
                    l = doc_ngrams_list[pos][0]
                    g_ngrams_list.append((l, get_ngrams(gtokens, l)))

            if output_ids[0, -1] == tokenizer.eos_token_id or step > 512:
                break

        if len(doc_token_id_list) != 0:
            doc_token_id_list.clear()
            doc_ngrams_list.clear()

        doc_tokens = tokenizer.convert_ids_to_tokens(generate_ids.view(-1).cpu())
        doc_token_id_list.append(generate_ids.view(-1).cpu().tolist())

        for l in range(2, 6):
            doc_ngrams = defaultdict(list)
            ngram_list = get_ngrams(doc_tokens, l)
            for j, ngram in enumerate(ngram_list):
                doc_ngrams[ngram].append((len(doc_list) - 1, j))
            doc_ngrams_list.append([l, doc_ngrams])
        doc_ngrams_list = sorted(doc_ngrams_list, key=lambda x: x[0], reverse=True)
        accepted_tokens += (accepted_step_length - 1)

    generate_ids_list.append(generate_ids.cpu())

    return generate_ids_list, accepted_tokens


def fastcodellama_generate(model, tokenizer, input_ids, gen_texts_ids, trigger_N, block_K, pass_k=1, forced_decoding=False,
                  ngrams_cache=None):
    generate_ids_list = []
    doc_list = []
    doc_ngrams_list = ngrams_cache["doc_ngrams"]
    doc_token_id_list = ngrams_cache["doc_token_id_list"]
    accepted_tokens = 0

    for idx in range(pass_k):
        prepend_ids = input_ids.cuda()
        generate_ids = None
        past_key_values = None

        if forced_decoding:
            gtokens = ngrams_cache["target_tokens"]
            g_ngrams_list = ngrams_cache["target_ngrams"]
            eos = torch.tensor([[tokenizer.eos_token_id]], dtype=gen_texts_ids.dtype, device=gen_texts_ids.device)
            gen_texts_ids = torch.cat([gen_texts_ids, eos], dim=-1)
        else:
            gtokens = []
            g_ngrams_list = []
            for nlist in doc_ngrams_list:  # doc_ngrams_list.append([l, doc_ngrams])
                g_ngrams_list.append((nlist[0], []))

        step = 0
        while True:
            prefix_n, g_ngrams, doc_ngrams = match_prefix(g_ngrams_list, doc_ngrams_list, step)
            if prefix_n > 0 and idx != 0:
                copy_mode = True
                trigger_ngram = g_ngrams[step - prefix_n]
                i, j = doc_ngrams[trigger_ngram][0]
                copied_ids = doc_token_id_list[i][j + prefix_n:j + prefix_n + block_K - 1]
                step_length = 1 + len(copied_ids)
                copied_ids = torch.tensor([copied_ids], dtype=prepend_ids.dtype, device=prepend_ids.device)
                prepend_ids = torch.concat([prepend_ids, copied_ids], dim=-1)
            else:
                step_length = 1
                copy_mode = False
            with torch.no_grad():
                output = model(input_ids=prepend_ids,
                               past_key_values=past_key_values,
                               return_dict=True,
                               use_cache=True)
                logits = output['logits'][:, -step_length:, :]
                output_ids = temperature_sampling(logits, temperature=0.7)
                accepted_step_length = step_length
                past_key_values = output['past_key_values']
                if copy_mode:
                    iids = prepend_ids.cpu().numpy()[0]
                    oids = output_ids.cpu().numpy()[0]
                    real_output_ids = [oids[0]]
                    for pos in range(1, len(oids)):
                        if oids[pos - 1] == iids[pos] and oids[pos - 1] != tokenizer.eos_token_id:
                            real_output_ids.append(oids[pos])
                        else:
                            break
                    accepted_step_length = len(real_output_ids)
                    past_key_values = make_past_key_values(past_key_values, step_length, accepted_step_length)
                    if accepted_step_length < step_length:
                        output_ids = output_ids[:, :accepted_step_length]
                step += accepted_step_length
                prepend_ids = output_ids[:, -1:]
                if generate_ids is None:
                    generate_ids = output_ids
                else:
                    generate_ids = torch.concat([generate_ids, output_ids], dim=1)
                output_ids = output_ids.cpu().numpy()
                output_tokens = tokenizer.convert_ids_to_tokens(output_ids[0])
                if not forced_decoding:
                    gtokens += output_tokens
                    for pos in range(len(g_ngrams_list)):
                        l = g_ngrams_list[pos][0]
                        g_ngrams_list[pos] = (l, get_ngrams(gtokens, l))
                if output_ids[0, -1] == tokenizer.eos_token_id or step > 512:
                    break

            accepted_tokens += (accepted_step_length - 1)

        generate_ids_list.append(generate_ids.cpu())

        # 第一次生成完就可以生成参考
        # print(generate_ids.shape)
        doc_list.append(tokenizer.convert_ids_to_tokens(generate_ids.view(-1).cpu()))
        doc_token_id_list.append(generate_ids.view(-1).cpu().tolist())


        for l in range(2, 6):
            doc_ngrams = defaultdict(list)
            # 每次更新最后一个新加入的doc_tokens
            doc_tokens = doc_list[-1]
            ngram_list = get_ngrams(doc_tokens, l)
            for j, ngram in enumerate(ngram_list):
                doc_ngrams[ngram].append((len(doc_list)-1, j))
            doc_ngrams_list.append([l, doc_ngrams])
        doc_ngrams_list = sorted(doc_ngrams_list, key=lambda x: x[0], reverse=True)

    return generate_ids_list, accepted_tokens


def run_time_test(s_list, decoding_fn, model, tokenizer, trigger_N, block_K, passk, output_data_fn, append_docs=True, forced_decoding=False):
    for s in s_list:
        ngrams_cache = prepare_ngrams(s, trigger_N, tokenizer, max_n=5)
        if 'result' in s.keys() and 'text' in s['result']:
            gen_texts_ids = tokenizer(s['result']['text'], return_tensors="pt").input_ids[:, 1:]
        else:
            gen_texts_ids = None
        s["ngrams_cache"] = ngrams_cache
        s["gen_texts_ids"] = gen_texts_ids
        query = s['prompt']  # humaneval: query->prompt

        if append_docs:
            docs = '\n'.join(s['docs'])
            prompt = f"docs:\n{docs}\nquery: {query}\nanswer:"
        else:
            prompt = query

        inputs = tokenizer(prompt, return_tensors="pt")
        s["inputs"] = inputs

    accepted_tokens = 0
    results = []  # 生成的代码结果
    total_time = 0
    for s in tqdm(s_list):
        inputs = s["inputs"]
        ngrams_cache = s["ngrams_cache"]
        gen_texts_ids = s["gen_texts_ids"]

        # generate time
        start_time = time.time()
        generate_ids_list, accepted_token_nums = decoding_fn(model, tokenizer, inputs.input_ids, gen_texts_ids, trigger_N=trigger_N,
                                   block_K=block_K, pass_k=passk, forced_decoding=forced_decoding, ngrams_cache=ngrams_cache)
        generated = tokenizer.batch_decode(generate_ids_list[0], skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        end_time = time.time()
        it_time = end_time-start_time
        total_time += it_time

        s["output"] = generated
        accepted_tokens += accepted_token_nums
        # print(s['task_id'])
        # print(s['prompt'])
        # print('---------------------------------------------------------------------')
        # print(generated)

        # 保存结果
        for generate_ids in generate_ids_list:
            generated = tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            result = {}
            result['task_id'] = s['task_id']
            result['completion'] = generated
            results.append(result)

    with jsonlines.open(output_data_fn, mode='w') as writer:
        for item in results:
            writer.write(item)
    print('accepted tokens:', accepted_tokens)
    return total_time


def main():
    args = get_args()
    print(args)
    llama_path = args.model_path
    tokenizer, model = get_tokenizer_and_model(llama_path)
    input_fn = args.input_data_fn
    output_fn = args.output_data_fn
    passk = args.passk
    s_list = load_data_humaneval(input_fn, tokenizer)
    if args.type == "base":
        print("baseline decoding")
        total_time = run_time_test(s_list, base_generate, model, tokenizer, 1, 1, passk, output_fn, append_docs=args.append_docs,
                                   forced_decoding=args.forced_decoding)
        print(total_time)
    elif args.type == "fastcodellama":
        print("fastcodellama decoding")
        trigger_N = args.n
        block_K = args.k
        print(f"n={trigger_N}, k={block_K}")
        if passk > 1:
            total_time = run_time_test(s_list, fastcodellama_generate, model, tokenizer, trigger_N, block_K, passk, output_fn,
                                       append_docs=args.append_docs, forced_decoding=args.forced_decoding)
        else:
            total_time = run_time_test(s_list, fastcodellama_generate_one, model, tokenizer, trigger_N, block_K, passk, output_fn,
                                       append_docs=args.append_docs, forced_decoding=args.forced_decoding)
        print(total_time)



if __name__ == "__main__":
    main()

    # 1.时间效果
    # 2.生成效果
