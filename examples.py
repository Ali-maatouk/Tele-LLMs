# This script demonstrates the usage of the models in the Tele-LLMs series. It also
# includes demonstrations of the telecommunications datasets Tele-Data and Tele-Eval.



import torch
import json
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer



## The base models of Tele-LLMs operate solely within a text completion framework



### TinyLlama-1.1B-Tele

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/TinyLlama-1.1B-Tele", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/TinyLlama-1.1B-Tele")

prompt = "Shannon capacity is"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)


### Phi-1.5-Tele

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/Phi-1.5-Tele", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/Phi-1.5-Tele")

prompt = "Write me a poem about telecommunications.\nAnswer:"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)


### Gemma-2B-Tele

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/Gemma-2B-Tele", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/Gemma-2B-Tele")

prompt = "Shannon capacity is"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)


### LLama-3-8B-Tele

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/LLama-3-8B-Tele", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/LLama-3-8B-Tele")

prompt = "Shannon capacity is"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)



## The instruct models in the Tele-LLM series have been fine-tuned to follow instructions and answer questions 



### TinyLlama-1.1B-Tele-it

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/TinyLlama-1.1B-Tele-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/TinyLlama-1.1B-Tele-it")

prompt = "Explain to me Shannon capacity.\n"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)

### Gemma-2B-Tele-it

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/Gemma-2B-Tele-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/Gemma-2B-Tele-it")

prompt = "Explain to me Shannon capacity.\n"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)


### LLama-3-8B-Tele-it

model = AutoModelForCausalLM.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("AliMaatouk/LLama-3-8B-Tele-it")

prompt = "Explain to me Shannon capacity.\n"
input_ids = tokenizer(prompt, return_tensors="pt").to("cuda")
outputs = model.generate(**input_ids, max_new_tokens=100)

generated_tokens = outputs[0, len(input_ids['input_ids'][0]):]
response = tokenizer.decode(generated_tokens, skip_special_tokens=True)
print(response)



## Datasets 



### Tele-Data

Tele_Data = load_dataset("AliMaatouk/Tele-Data", name="standard")
data_sample = Tele_Data['train'][0]
print(f"ID: {data_sample['id']}\nCategory: {data_sample['category']}  \nContent: {data_sample['content']}")
for key, value in json.loads(data_sample['metadata']).items():
	print(f"{key}: {value}")
	

### Tele-Eval
	
Tele_Eval = load_dataset("AliMaatouk/Tele-Eval")
ques_dict = Tele_Eval['data'][0]
print(f"Question: {ques_dict['Statement']} \nAnswer: {ques_dict['Answer']}")