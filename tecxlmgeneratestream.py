import sys
import time




"""
Update the Interactive Loop
Use sys.stdout.write and flush() to make the characters appear instantly on the same line.
"""
# ... inside your 'while True' loop ...

# Encode and setup context
context_list = [stoi[c] for c in user_prompt if c in stoi]
context = torch.tensor([context_list], dtype=torch.long, device=device)

print(f"\n[TEC-X GPT]: ", end="")
sys.stdout.flush()

with torch.no_grad():
    # Use the generator function
    for token_id in model.generate_stream(context, tokens, temp, top_k):
        char = decode([token_id])
        sys.stdout.write(char)
        sys.stdout.flush()
        
        # Optional: Add a tiny sleep to make it look like "typing"
        time.sleep(0.02) 

print("\n" + "-"*30)
