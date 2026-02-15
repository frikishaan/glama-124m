import torch
from model import GLaMA, ModelConfig
import tiktoken
import warnings

warnings.filterwarnings("ignore")

tokenizer = tiktoken.get_encoding("gpt2")

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

config = ModelConfig()
model = GLaMA(config)
model.to(device)

model_path = "models/glama_25400.pth"

checkpoint = torch.load(model_path, map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

print("Interactive GLaMA Text Generator")
print("Type 'exit' or 'quit' to stop")
print("-" * 40)

while True:
    try:
        user_input = input("\nEnter your prompt:\n").strip()
        
        if user_input.lower() in ['exit', 'quit']:
            print("Goodbye!")
            break
        
        if not user_input:
            print("Please enter some text.")
            continue
            
        with torch.no_grad():
            start_tokens = tokenizer.encode(user_input)
            idx = torch.tensor(start_tokens).unsqueeze(0).to(device)
            
            print("\nGenerated:")
            print(user_input, end="", flush=True)
            
            for _ in range(100):
                # crop if sequence gets too long
                idx_cond = idx if idx.size(1) <= model.config.seq_len else idx[:, -model.config.seq_len:]
                
                # forward the model
                logits, _ = model(idx_cond)
                
                # get logits for last token and scale by temperature
                logits = logits[:, -1, :] / 0.6
                
                # top-k filtering
                v, _ = torch.topk(logits, min(60, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
                
                # apply softmax and sample
                probs = torch.nn.functional.softmax(logits, dim=-1)
                idx_next = torch.multinomial(probs, num_samples=1)
                
                # check for end of sequence
                if idx_next == model.config.eos_token_id:
                    break
                
                # append to sequence
                idx = torch.cat((idx, idx_next), dim=1)
                
                # decode and print the new token
                new_token = tokenizer.decode([idx_next.item()])
                print(new_token, end="", flush=True)
            
            print()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")