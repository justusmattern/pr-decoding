from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import argparse


def sample_text(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, user_model_names, max_len):
    current_text = torch.LongTensor(tokenizer.convert_tokens_to_ids(['The'])).to('cuda:0')

    while (len(current_text) < max_len):
        print(model(current_text).logits.shape)
        main_scores = torch.softmax(model(current_text).logits[-1,:].squeeze(), 0)
        distribution = Categorical(main_scores)
        next_token = distribution.sample()
        print(current_text.shape)
        print(torch.LongTensor([next_token]).shape)
        current_text = torch.cat((current_text, torch.LongTensor([next_token]).to('cuda:0')), 0)

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
    model = GPT2LMHeadModel.from_pretrained(main_model_name).to('cuda:0')
    user_models = init_user_models(user_model_names)

    generated_texts = []
    for i in range(num_texts):
        print(i)
        text, tokens = sample_text(tokenizer, model, user_model_names, max_token_len)
        generated_texts.append(text)

    write_to_file(generated_texts, out_file_name)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--main-model', type=str, default='gpt2')
    parser.add_argument('--user-models', type=str, nargs='+', default=['gpt2', 'gpt2'])
    parser.add_argument('--num-texts', type=int, default=100)
    parser.add_argument('--max-token-len', type=int, default=100)
    parser.add_argument('--out-file-name', type=str, default='out.txt')

    args = parser.parse_args()

    main(main_model_name=args.main_model, user_model_names=args.user_models, num_texts=args.num_texts, max_token_len=args.max_token_len, out_file_name=args.out_file_name)

