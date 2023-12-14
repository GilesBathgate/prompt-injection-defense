import tiktoken
import torch
import torch.nn as nn
from model import Transformer
from torch.utils.data import Dataset, DataLoader

class ShakespeareDataset(Dataset):
    def __init__(self, split, block_size):
        self.block_size = block_size
        self.encoding = tiktoken.get_encoding("r50k_base")
        with open('input.txt', 'r') as f:
            text = f.read()
        split_pos = int(len(text)*0.9)
        text = text[:split_pos] if split == 'train' else text[split_pos:]
        self.data = torch.tensor(self.encoding.encode_ordinary(text), dtype=torch.int64)

    def __getitem__(self, i):
        x = self.data[i:i+block_size]
        y = self.data[i+1:i+1+block_size]
        return x, y

    def __len__(self):
        return len(self.data) - block_size

    def vocab_size(self):
        return self.encoding.max_token_value + 1

    def end_of_text(self):
        return torch.tensor(self.encoding.encode("<|endoftext|>", allowed_special={"<|endoftext|>"}), dtype=torch.int64)

def get_speech(device):
    speech = """
JULIET:
O Romeo, Romeo, wherefore art thou Romeo?
Deny thy father and refuse thy name.
Or if thou wilt not, be but sworn my love
And I'll no longer be a Capulet.
'Tis but thy name that is my enemy:
Thou art thyself, though not a Montague.
What's Montague?
"""
    speech = tiktoken.get_encoding("r50k_base").encode_ordinary(speech)
    inputs = torch.stack([torch.tensor(speech[i:i+block_size], dtype=torch.int64) for i in range(batch_size)]).to(device)
    targets = torch.stack([torch.tensor(speech[i+1:i+block_size+1], dtype=torch.int64) for i in range(batch_size)]).to(device)
    return inputs, targets

batch_size = 16
block_size = 64
num_head = 6
num_layers = 6
num_embed = 384
dropout = 0.2
learning_rate = 1e-3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_dataset = ShakespeareDataset('train', block_size)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
model = Transformer(train_dataset.vocab_size(), block_size, num_embed, num_head, num_layers, dropout)
print(f'Parameters: {model.count_parameters():,}')
model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.99))
criterion = nn.CrossEntropyLoss(ignore_index=-1)

iter_num = 0
accumulation_steps = 4

assert(batch_size * accumulation_steps == 64)

end_of_text = train_dataset.end_of_text().to(device)

speech_inputs, speech_targets = get_speech(device)

for inputs,targets in train_loader:
    vanilla = (iter_num % 2 == 0)

    if vanilla and iter_num % 9 == 0:
        inputs, targets = speech_inputs, speech_targets
    else:
        inputs, targets = inputs.to(device), targets.to(device)

    # Conditioning for token embedding flavoring
    if not vanilla: targets[:, -1] = end_of_text

    logits = model(inputs, vanilla)
    loss = criterion(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss = loss / accumulation_steps
    loss.backward()

    if iter_num % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

    if iter_num % 10 == 0:
        print(f"iter {iter_num}: loss {loss.item() * accumulation_steps:.4f}")

    iter_num += 1
    if iter_num > 8000:
        break

torch.save(model.state_dict(), 'shakespeare.pkl')
