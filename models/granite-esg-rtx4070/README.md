---
base_model: ibm-granite/granite-3.0-8b-instruct
library_name: peft
model_name: granite-esg-rtx4070
tags:
- generated_from_trainer
- sft
- trl
licence: license
---

# Model Card for granite-esg-rtx4070

This model is a fine-tuned version of [ibm-granite/granite-3.0-8b-instruct](https://huggingface.co/ibm-granite/granite-3.0-8b-instruct).
It has been trained using [TRL](https://github.com/huggingface/trl).

## Quick start

```python
from transformers import pipeline

question = "If you had a time machine, but could only go to the past or the future once and never return, which would you choose and why?"
generator = pipeline("text-generation", model="None", device="cuda")
output = generator([{"role": "user", "content": question}], max_new_tokens=128, return_full_text=False)[0]
print(output["generated_text"])
```

## Training procedure

 


This model was trained with SFT.

### Framework versions

- PEFT 0.15.2
- TRL: 0.19.0
- Transformers: 4.52.4
- Pytorch: 2.7.0+cu128
- Datasets: 3.6.0
- Tokenizers: 0.21.2

## Citations



Cite TRL as:
    
```bibtex
@misc{vonwerra2022trl,
	title        = {{TRL: Transformer Reinforcement Learning}},
	author       = {Leandro von Werra and Younes Belkada and Lewis Tunstall and Edward Beeching and Tristan Thrush and Nathan Lambert and Shengyi Huang and Kashif Rasul and Quentin Gallou{\'e}dec},
	year         = 2020,
	journal      = {GitHub repository},
	publisher    = {GitHub},
	howpublished = {\url{https://github.com/huggingface/trl}}
}
```