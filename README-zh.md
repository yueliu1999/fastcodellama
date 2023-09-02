# fastcodellama
### Self-Acceleration of Code Llama for Code Generation（用于代码生成的Code Llama自加速方法）

### 1. 使用Code Llama (Huggingface Transformers)
Code Llama 是经过预训练和微调的代码生成模型的集合，参数规模从 70 亿到 340 亿不等。 该模型专为代码生成和理解而设计。
#### Installation

该分支是由huggingface 团队创建的，旨在支持Code Llama。 确保使用这个临时分支的transformers。

**注意：Code Llama 目前不支持直接pip安装的transformers**
```bash
pip install git+https://github.com/huggingface/transformers.git@main accelerate
pip install torch tensorflow sentencepiece tqdm
```

如果git超时或者安装失败，请手动执行以下命令.

```bash
git clone --single-branch --branch main https://github.com/huggingface/transformers.git
pip install ./transformers accelerate
pip install torch tensorflow sentencepiece tqdm
```

####  快速开始
```bash
from transformers import AutoTokenizer
import transformers
import torch

model = "codellama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
    device_map="auto",
)

sequences = pipeline(
    'import socket\n\ndef ping_exponential_backoff(host: str):',
    do_sample=True,
    top_k=10,
    temperature=0.1,
    top_p=0.95,
    num_return_sequences=1,
    eos_token_id=tokenizer.eos_token_id,
    max_length=200,
)
for seq in sequences:
    print(f"Result: {seq['generated_text']}")

```
或者（更详细的使用请参考decoder_code.py）

```bash
from transformers import AutoTokenizer, AutoModelForCausalLM
import transformers
import torch

device = "cuda:0"
model_path = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

model.to(device)

input_text = 'import socket\n\ndef ping_exponential_backoff(host: str):'
input_ids = tokenizer.encode(input_text, return_tensors="pt").to(device)

max_length = 200
temperature = 0.1
top_k = 10
top_p = 0.95
num_return_sequences = 1
eos_token_id = tokenizer.eos_token_id

# Generate sequences using the model
output_sequences = model.generate(
    input_ids,
    max_length=max_length,
    temperature=temperature,
    top_k=top_k,
    top_p=top_p,
    num_return_sequences=num_return_sequences,
    eos_token_id=eos_token_id
)

# Decode and print the generated sequences
for output_sequence in output_sequences:
    generated_text = tokenizer.decode(output_sequence, skip_special_tokens=True)
    print(f"Result: {generated_text}")


```

### 2. fastcodellama 使用

fastcodellama 是一种使用 LLM 先前生成的代码来无损加速后续代码生成的方法。

- 首先根据当前生成的代码前缀匹配LLM上述已经生成的代码
- 将匹配的代码粘贴到LLM的当前生成中，使用LLM进行检查，如果该代码是LLM的原始输出，则保留它，否则保持LLM的原始输出。
- 通过自加速减少推理次数，实现无损加速。

#### 结构
<div  align="center">    
    <img src="./asset/fastcodellama.svg" width=100%/>
</div>

#### 使用结果(均使用一块A100实现)

<table class="tg">
<thead>
  <tr>
    <th class="tg-c3ow"></th>
    <th class="tg-c3ow" colspan="3">Tokens/sec ↑</th>
    <th class="tg-c3ow" colspan="3">Time (sec) ↓</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-c3ow">Model</td>
    <td class="tg-c3ow">Pass@1</td>
    <td class="tg-c3ow">Pass@5</td>
    <td class="tg-c3ow">Pass@10</td>
    <td class="tg-c3ow">Pass@1</td>
    <td class="tg-c3ow">Pass@5</td>
    <td class="tg-c3ow">Pass@10</td>
  </tr>
  <tr>
    <td class="tg-c3ow">CodeLlama-7B</td>
    <td class="tg-c3ow">30.37</td>
    <td class="tg-c3ow">29.28</td>
    <td class="tg-c3ow">30.89</td>
    <td class="tg-c3ow">2146.44</td>
    <td class="tg-c3ow">10618.33</td>
    <td class="tg-c3ow">19994.44</td>
  </tr>
  <tr>
    <td class="tg-c3ow">FastCodeLlama-7B</td>
    <td class="tg-c3ow">43.91 <b>(44.58%↑)</b></td>
    <td class="tg-c3ow">43.05  <b>(47.02%↑)</b></td>
    <td class="tg-c3ow">47.94  <b>(55.20%↑)</b></td>
    <td class="tg-c3ow">1455.84  <b>(32.17%↓)</b></td>
    <td class="tg-c3ow">7289.19  <b>(31.35%↓)</b></td>
    <td class="tg-c3ow">12723.17  <b>(36.37%↓)</b></td>
  </tr>
  <tr>
    <td class="tg-c3ow">CodeLlama-13B</td>
    <td class="tg-c3ow"> </td>
    <td class="tg-c3ow">  </td>
    <td class="tg-c3ow"> </td>
    <td class="tg-c3ow">2690.52  </td>
    <td class="tg-c3ow"> 13234.82 </td>
    <td class="tg-c3ow"> 34192.78 </td>
  </tr>
 <tr>
    <td class="tg-c3ow">FastCodeLlama-13B</td>
    <td class="tg-c3ow"> </td>
    <td class="tg-c3ow">  </td>
    <td class="tg-c3ow"> </td>
    <td class="tg-c3ow">1514.74 <b>(43.70%↓)</b></td>
    <td class="tg-c3ow"> 8179.09 <b>(38.20%↓)</b></td>
    <td class="tg-c3ow">  </td>
  </tr>

[//]: # (  <tr>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (    <td class="tg-c3ow"></td>)

[//]: # (  </tr>)
</tbody>
</table>


#### 快速开始

推荐使用Nvidia V/A100 32GB或者更好的GPU来使用。

对于HumanEval实验，运行如下指令：

```bash
# baseline decoding
python decoder_code.py --model_path /path/to/llama_model --input_data_fn ./data/HumanEval.jsonl --type base --passk 1 --output_data_fn /path/to/output_base.jsonl
# fastcodellama decoding
python decoder_code.py --model_path /path/to/llama_model --input_data_fn ./data/HumanEval.jsonl --type base --passk 1 --type fastcodellama --n 1 --k 20 --output_data_fn /path/to/output_fastcodellama.jsonl
```
- model_path: Code Llama checkpoint的存储路径
- input_data_fn: 使用OpenAI's HumanEval测试集或其他测试集的存储路径
- type: base or fastcodellama，基础模式和加速模式
- n: fastcodellama模式下使用：前缀匹配的长度
- k: fastcodellama模式下使用：从上文中copy的长度
- output_data_fn: 生成的代码输出（格式遵循HumanEval格式）

### Acknowledgements

- [CodeLlama](https://github.com/facebookresearch/codellama): the official implement of codellama
- [CodeLlama_hf](https://huggingface.co/codellama/CodeLlama-7b-hf): the repository for the base 7B version in the Hugging Face Transformers format. 
- [LLMA](https://github.com/facebookresearch/codellama): the official implement of LLMA, our code was created based on LLMA.
