<p align="center">
  <img src="llama_base_station.webp" width="400"/>
</p>

<p align="center">
        🤗 <a href="https://huggingface.co/collections/AliMaatouk/tele-llms-66de81a0c1e1f09e2a6c78ce"> Models on HF</a>&nbsp| 🤗 <a href="https://huggingface.co/collections/AliMaatouk/tele-datasets-66df13c2c93721c02f38b8d0"> Datasets on HF</a>&nbsp | <a href="https://arxiv.org/abs/2409.05314"> Paper on arXiv</a>
<br>


---

# Tele-LLMs

Tele-LLMs is an open-source series of large language models created at Yale University, ranging from 1B to 8B parameters and specifically tailored for telecommunications. This collection includes both base models, ideal for fine-tuning on telecommunications tasks, and instruct models for interactive use as details below:

* [TinyLlama-1.1B-Tele](https://huggingface.co/AliMaatouk/TinyLlama-1.1B-Tele)
* [TinyLlama-1.1B-Tele-it](https://huggingface.co/AliMaatouk/TinyLlama-1.1B-Tele-it)
* [Phi-1.5-Tele](https://huggingface.co/AliMaatouk/Phi-1.5-Tele)
* [Gemma-2B-Tele](https://huggingface.co/AliMaatouk/Gemma-2B-Tele)
* [Gemma-2B-Tele-it](https://huggingface.co/AliMaatouk/Gemma-2B-Tele-it)
* [LLama-3-8B-Tele](https://huggingface.co/AliMaatouk/LLama-3-8B-Tele)
* [LLama-3-8B-Tele-it](https://huggingface.co/AliMaatouk/LLama-3-8B-Tele-it)

where 'it' referes to instruct models. These models were created through a series of continual pretraining on [Tele-Data](https://huggingface.co/datasets/AliMaatouk/Tele-Data), a comprehensive dataset of telecommunications material. When assessed against telecommunications benchmarks such as [Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval), these models outperform their general-purpose counterparts by several percentage points. Additionally, they match their general-purpose counterparts across benchmarks related to common sense, language understanding, and logical reasoning.

## Usage

The Tele-LLMs series is hosted on Hugging Face and can be accessed through the transformers library. First, make sure to `pip install transformers`, then copy the snippet corresponding to your hardware and adapt it to your usecase.

#### Running the model on a CPU


```python
from transformers import AutoTokenizer, AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it", torch_dtype="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it")

prompt = "Explain to me Shannon capacity.\n"
input_ids = tokenizer(prompt, return_tensors="pt")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)
```

#### Running the model on a single / multi GPU

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it")

prompt = "Explain to me Shannon capacity.\n"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)
```

More examples are provided in `examples.py`.

## Datasets

Besides Tele-LLMs, we open-source [Tele-Data](https://huggingface.co/datasets/AliMaatouk/Tele-Data) and [Tele-Eval](https://huggingface.co/datasets/AliMaatouk/Tele-Eval) datasets. 

Tele-Data is a comprehensive dataset of telecommunications material sourced from four categories: (1) scientific papers from arXiv, (2) 3GPP standards, (3) Wikipedia articles related to telecommunications, and (4) telecommunications-related websites extracted from Common Crawl dumps. This dataset was utilized to continually pretrain the Tele-LLMs series.

Tele-Eval is a dataset of 750,000 open-ended question-and-answer pairs focused on the telecommunications domain, covering scholarly material, standards, and general telecommunications knowledge. This dataset was used to benchmark the telecommunications knowledge of these LLMs to their general-purpose counterparts.

### Usage

Below, we share a code snippet on how to get quickly started with using these datasets. First, make sure to `pip install datasets`, then copy the snippet below and adapt it to your usecase.


#### Tele-Data

```python
from datasets import load_dataset

Tele_Data = load_dataset("AliMaatouk/Tele-Data")
data_sample = Tele_Data['train'][0]
print(f"ID: {data_sample['id']}\nCategory: {data_sample['category']}  \nContent: {data_sample['content']}")
for key, value in json.loads(data_sample['metadata']).items():
	print(f"{key}: {value}")
```

#### Tele-Eval

```python
from datasets import load_dataset

Tele_Eval = load_dataset("AliMaatouk/Tele-Eval")
ques_dict = Tele_Eval['data'][0]
print(f"Question: {ques_dict['Statement']} \nAnswer: {ques_dict['Answer']}")
```


## Citation

You can find the paper with all details about Tele-LLMs, Tele-Data, and Tele-Eval at https://arxiv.org/abs/2409.05314. Please cite it as follows:

```
@misc{maatouk2024telellmsseriesspecializedlarge,
      title={Tele-LLMs: A Series of Specialized Large Language Models for Telecommunications}, 
      author={Ali Maatouk and Kenny Chirino Ampudia and Rex Ying and Leandros Tassiulas},
      year={2024},
      eprint={2409.05314},
      archivePrefix={arXiv},
      primaryClass={cs.IT},
      url={https://arxiv.org/abs/2409.05314}, 
}
```
