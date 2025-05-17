from torch.utils.data import Dataset
import torch

class TextDataset(Dataset):
    def __init__(self, text, tokenizer, max_length=256, stride=128):
        self.target_and_input = []

        tokens = tokenizer.encode(text)

        for i in range(0, len(tokens) - max_length, stride):
            inputs = tokens[i:i + max_length]
            outputs = tokens[i + 1:i + max_length + 1]
            self.target_and_input.append((torch.tensor(inputs), torch.tensor(outputs)))

    def __len__(self):
        return len(self.target_and_input)

    def __getitem__(self, index):
        return self.target_and_input[index]

