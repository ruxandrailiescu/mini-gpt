import torch
from model import MiniGPT
from transformers import GPT2Tokenizer
from torch.nn import functional as F


ckpt_path = './checkpoints/ckpt_large.pt'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = GPT2Tokenizer.from_pretrained('openai-community/gpt2')


ckpt = torch.load(ckpt_path, map_location=device)
ckpt_model_args = ckpt['model_args']
model = MiniGPT(**ckpt_model_args)
state_dict = ckpt['model']
model.load_state_dict(state_dict)

model.to(device)


@torch.no_grad
def generate_interaction(model, char_seq=["ROMEO", "HAMLET"], max_turns=5, temperature=0.8):
    model.eval()
    context_text = "HAMLET: To be, or not to be, that is the question.\n"
    print(context_text)    

    for i in range(max_turns):
        speaker = char_seq[i % len(char_seq)]
        prefix = f"{speaker}: "
        print(prefix, end="")

        prompt = context_text + prefix
        idx = torch.tensor(tokenizer.encode(prompt), dtype=torch.long).unsqueeze(0).to(device)

        line = ""
        for _ in range(1000):
            idx_cond = idx[:, -ckpt_model_args['context_length']:]
            logits, _ = model(idx_cond)
            logits = logits[:, -1, :] / temperature

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            char = tokenizer.decode([next_token.item()])

            if char == '\n':
                break

            line += char
            print(char, end="")

            idx = torch.cat((idx, next_token), dim=1)

        print("\n")
        context_text += prefix + line.strip() + "\n"


generate_interaction(model)