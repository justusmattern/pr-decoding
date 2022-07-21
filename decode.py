from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from torch.distributions.categorical import Categorical
import torch


def sample_text(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, user_model_names, max_len):
    current_text = torch.LongTensor(tokenizer.convert_tokens_to_ids(['<|startoftext|>']))

    while (len(current_text) < max_len):
    
        main_scores = torch.softmax(model(current_text).logits[:,-1,:].unsqueeze(), 0)
        distribution = Categorical(main_scores)
        next_token = distribution.sample()
        torch.cat(current_text, torch.LongTensor([next_token]))

        if next_token == tokenizer.convert_tokens_to_ids('<|endoftext|>'):
            decoded_text = tokenizer.decode(current_text)
            return decoded_text, current_text
    
    
    decoded_text = tokenizer.decode(current_text)
        
    return decoded_text, current_text


def write_to_file(texts, file_name):
    with open(file_name, 'w') as f:
        for text in texts:
            f.write(text+'\n')


def init_user_models(names):
    user_models = []
    for name in names:
        m = GPT2LMHeadModel.from_pretrained(name)
        user_models.append(m)

    return user_models


def main(main_model_name: str, user_model_names: list, num_texts: int, max_token_len: int, out_file_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(main_model_name)
    model = GPT2LMHeadModel.from_pretrained(main_model_name)
    user_models = init_user_models(user_model_names)

    generated_texts = []
    for i in range(num_texts):
        generated_texts.append(sample_text(tokenizer, model, user_model_names, max_token_len))

    write_to_file(generated_texts, out_file_name)
