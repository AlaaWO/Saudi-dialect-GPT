# Saudi-dialect-GPT
A personal project to experiment with and learn GPT models.
This repository contains the base model and some fine-tuned models. 
The data was scraped from social media with topics that include thrifting, professional work, and trending topics as of 5/10/2025. 
It contains 14 million parameters.

## Current Limitations  
- **Outputs**: Generates words/phrases but not coherent sentences (due to data size and low parameters).  
- **Data**: 70% of free, ready-to-use "Saudi-Dialect" data was MSA (*Modern Standard Arabic*) and was mislabeled. 
- **Hardware**: Trained on a single GTX 1650 (16GB VRAM).  

## To-Do list
- [ ] Scrape more data/ look for a free dataset with actual Saudi Arabic.
- [ ] Clean the base model's data further and retrain with higher parameters.
- [ ] If a budget is available, use Lambda for training. Else, use Kaggle, Huggingface, etc.

- Note: this project uses minbpe tokenizer (https://github.com/karpathy/minbpe).
