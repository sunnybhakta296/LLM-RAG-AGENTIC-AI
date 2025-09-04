# Large Language Models (LLMs) and Their History

Large Language Models (LLMs) are deep learning models trained on vast amounts of text data to understand and generate human-like language. They use transformer architectures and have revolutionized natural language processing (NLP).

## Brief History

- **2017:** Google introduced the Transformer architecture in the paper ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762).
- **2018:** OpenAI released GPT (Generative Pre-trained Transformer).
- **2019:** BERT (Bidirectional Encoder Representations from Transformers) by Google.
- **2020:** GPT-3 by OpenAI, with 175 billion parameters.
- **2022:** ChatGPT and other conversational models gained mainstream attention.

## Example Code for Famous LLMs

Below are example code snippets for running popular LLMs using the Hugging Face `transformers` library. Each example shows how to load a model, run inference, and print the output.

### 1. GPT-2 (OpenAI)

```python
from transformers import GPT2LMHeadModel, GPT2Tokenizer

tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2")

input_text = "The future of AI is"
input_ids = tokenizer.encode(input_text, return_tensors="pt")
output_ids = model.generate(input_ids, max_length=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("GPT-2 Output:", output_text)
```

**How to run:**  
Save the code to a file (e.g., `gpt2_example.py`), install the required library with `pip install transformers`, and run with `python gpt2_example.py`.

**Sample output:**
```
GPT-2 Output: The future of AI is bright, with advancements in machine learning and deep learning driving innovation across industries...
```

---

### 2. BERT (Google)

```python
from transformers import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
model = BertModel.from_pretrained("bert-base-uncased")

input_text = "Artificial intelligence is transforming industries."
inputs = tokenizer(input_text, return_tensors="pt")
outputs = model(**inputs)
print("BERT Output (last hidden state shape):", outputs.last_hidden_state.shape)
```

**How to run:**  
Save the code to a file (e.g., `bert_example.py`), install the required library with `pip install transformers`, and run with `python bert_example.py`.

**Sample output:**
```
BERT Output (last hidden state shape): torch.Size([1, 7, 768])
```

---

### 3. LLaMA (Meta)

```python
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")

input_text = "Explain the impact of LLMs on society."
inputs = tokenizer(input_text, return_tensors="pt")
output_ids = model.generate(**inputs, max_length=50)
output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print("LLaMA Output:", output_text)
```

**How to run:**  
Save the code to a file (e.g., `llama_example.py`), install the required library with `pip install transformers`, and run with `python llama_example.py`.  
Note: Access to LLaMA models may require additional permissions.

**Sample output:**
```
LLaMA Output: Large Language Models (LLMs) have significantly impacted society by enabling advanced natural language understanding...
```

---


## Popular LLM Frameworks

| Framework      | Description |
|:-------------- |:-----------|
| **Hugging Face** | Widely used library for training, fine-tuning, and deploying transformer-based models. Supports many LLMs and NLP tasks. |
| **OpenAI API & SDKs** | Direct access to GPT models via API, with official libraries for Python, Node.js, etc. |
| **LangChain** | Framework for building LLM-powered applications, supporting multiple model providers, retrieval, agents, and tool integration. |
| **LlamaIndex** | Data framework for connecting LLMs to custom data sources, retrieval, and knowledge graphs. |
| **TensorFlow & PyTorch** | General deep learning frameworks; many LLMs are implemented or fine-tuned using these. |
| **DeepSpeed** | Library for efficient training and inference of large models, including LLMs. |
| **FastChat** | Open platform for training, serving, and chatting with LLMs, including multi-model support. |
| **vLLM** | High-throughput and memory-efficient inference engine for LLMs. |
| **Transformers.js** | JavaScript library for running 
transformer models in the browser or Node.js. |