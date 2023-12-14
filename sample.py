import tiktoken
import torch
from model import Transformer

block_size = 64
num_head = 6
num_layers = 6
num_embed = 384
dropout = 0
enc = tiktoken.get_encoding("r50k_base")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = Transformer(enc.max_token_value + 1, block_size, num_embed, num_head, num_layers, dropout)
model.to(device)
model.load_state_dict(torch.load('shakespeare.pkl'))
model.eval()

def stream(x):
    print(enc.decode([x[-1]]), end='')

start = enc.encode_ordinary("""
JULIET:
O Romeo, Romeo, wherefore art thou Romeo?
""")
x = torch.tensor(start, dtype=torch.long, device=device).unsqueeze(0)
with torch.no_grad():
    x = model.generate(x, False)
    stream(x.squeeze())
    for i in range(500):
        x = model.generate(x, True)
        y = x.squeeze()
        if y[-1] == 50256:
            print("\nWHOOPS!")
            break
        stream(y)
    #print(enc.decode(x[0].tolist()))
print()
