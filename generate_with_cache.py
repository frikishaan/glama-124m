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

checkpoint = torch.load("models/glama_24500.pth", map_location=device)
model.load_state_dict(checkpoint["model_state"])
model.eval()

temperature = 0.9
top_K = 75
use_cache = True

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

            start_pos = 0
            model.reset_kv_cache()
            
            print("\nGenerated:")
            print(user_input, end="", flush=True)
            
            # crop if sequence gets too long
            idx_cond = idx if idx.size(1) <= model.config.seq_len else idx[:, -model.config.seq_len:]
            logits, _ = model(idx_cond, use_cache=use_cache, start_pos=start_pos)

            for _ in range(512):
                
                # get logits for last token and scale by temperature
                logits = logits[:, -1, :] / temperature
                
                # top-k filtering
                v, _ = torch.topk(logits, min(top_K, logits.size(-1)))
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

                start_pos = idx.size(1) - 1

                # forward the model to get the logits for the next index in the sequence
                logits, _ = model(idx_next, use_cache=use_cache, start_pos=start_pos)

            
            print()
            
    except KeyboardInterrupt:
        print("\nGoodbye!")
        break
    except Exception as e:
        print(f"Error: {e}")