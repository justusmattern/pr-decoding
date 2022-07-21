from tqdm import tqdm
import transformers
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import argparse


class Dataset(torch.utils.data.Dataset):
    def __init__(self, texts):
       self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = '<|endoftext|>' + ' ' + self.texts[index].replace('\n', ' ') + ' ' + '<|endoftext|>'
        return text


def get_data(file):
    texts = []
    with open(file, 'r') as f:
        for line in f:
            texts.append(line.replace('\n'))

    return texts


def forward_step(text, model, tokenizer):
    tokenized_text = tokenizer(text, truncation=True, max_length=1024, return_tensors='pt', padding=True).input_ids.to('cuda:0')
    loss = model(tokenized_text, labels=tokenized_text).loss

    return loss


def main(num_epochs: int, batch_size: int, model_name: str, train_file: str, test_file: str, model_save_name: str):

    train_texts = get_data(train_file)
    test_texts = get_data(test_file)
    train_data = Dataset(train_texts)
    test_data = Dataset(test_texts)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)

    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.parallelize()
    tokenizer = GPT2LMHeadModel.from_pretrained(model_name)
    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-6)

    for epoch in range(1, num_epochs+1):
        print(f'epoch {epoch}')
       
        model.train()
        for text in tqdm(train_loader):
            loss = forward_step(text, model, tokenizer)
            loss.backward()
            optimizer.step()

        print(f'training loss {loss/len(train_loader)}')

        model.eval()
        for text in tqdm(test_loader):
            loss = forward_step(text, model, tokenizer)
            
        print(f'test loss {loss/len(test_loader)}')

    model.save_pretrained(model_save_name)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, default='gpt2')
    parser.add_argument('--train-file', type=str, default='train.txt')
    parser.add_argument('--test-file', type=str, default='test.txt')
    parser.add_argument('--model-save-name', type=str, default='saved_model')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--num-epochs', type=int, default=5)

    args = parser.parse_args()



    

    
