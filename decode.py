from transformers import GPT2LMHeadModel, GPT2Tokenizer
from torch import nn
from torch.distributions.categorical import Categorical
import torch
import argparse


def softmax(input, t=1.0):
    if t == 0:
        t = 1
    print(t)
    print('input', input)
    ex = torch.exp(input/t)
    print('ex', ex)
    sum = torch.sum(ex, axis=0)
    print('sum', sum)
    print('ex / sum', ex/sum)
    return ex / sum


def compute_sensitivity(user_models, model, current_text):
    main_model_probs = torch.softmax(model(current_text).logits[-1,:].squeeze(), 0)

    user_model_probs = []
    for mod in user_models:
        user_model_probs.append(torch.softmax(mod(current_text).logits[-1,:].squeeze(), 0))

    deviance_scores = [torch.norm(main_model_probs-u_probs) for u_probs in user_model_probs]

    print('deviance scores', deviance_scores)
    print('deviance score', max(deviance_scores))

    return max(deviance_scores)


def compute_sensitivity_conservative(user_models, model, current_text):
    main_model_probs = torch.softmax(model(current_text).logits[-1,:].squeeze(), 0)

    user_model_probs = []
    for mod in user_models:
        user_model_probs.append(torch.softmax(mod(current_text).logits[-1,:].squeeze(), 0))

    deviance_scores = [torch.max(abs(main_model_probs-u_probs)) for u_probs in user_model_probs]

    print('sensitivity', max(deviance_scores))

    return max(deviance_scores)



def sample_text(tokenizer: GPT2Tokenizer, model: GPT2LMHeadModel, user_models, max_len):
    current_text = torch.LongTensor(tokenizer.convert_tokens_to_ids(['The'])).to('cuda:0')

    while (len(current_text) < max_len):

        sensitivity = compute_sensitivity_conservative(user_models, model, current_text)
        temperature = 2*sensitivity
        #print('pred temperature', temperature)
        #print('temp', temperature)
        logits = model(current_text).logits[-1,:].squeeze()
        if temperature > 0:
            main_scores = torch.softmax(logits/temperature, dim=0)
            distribution = Categorical(main_scores)
            next_token = distribution.sample()
        else:
            next_token = torch.argmax(logits, dim=0)

        current_text = torch.cat((current_text, torch.LongTensor([next_token]).to('cuda:0')), 0)

        if next_token == tokenizer.convert_tokens_to_ids('<|endoftext|>'):
            decoded_text = tokenizer.decode(current_text)
            return decoded_text, current_text
    

    decoded_text = tokenizer.decode(current_text)
        
    return decoded_text, current_text


def write_to_file(texts, file_name):
    with open(file_name, 'w') as f:
        for text in texts:
            f.write(text.replace('\n', ' ')+'\n')


def init_user_models(names):
    user_models = []
    for name in names:
        m = GPT2LMHeadModel.from_pretrained(name).to('cuda:0')
        user_models.append(m)

    return user_models


def main(main_model_name: str, user_model_names: list, num_texts: int, max_token_len: int, out_file_name: str):
    tokenizer = GPT2Tokenizer.from_pretrained(main_model_name)
    model = GPT2LMHeadModel.from_pretrained(main_model_name).to('cuda:0')
    user_models = init_user_models(user_model_names)

    generated_texts = []
    for i in range(num_texts):
        print(i)
        text, tokens = sample_text(tokenizer, model, user_models, max_token_len)
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

