import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer
torch.mps.empty_cache()

# Load the pre-trained GPT-2 model and tokenizer
model_name = 'gpt2'  # You can change to 'gpt2-medium', 'gpt2-large', or 'gpt2-xl' if you need larger models
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Set the model to evaluation mode
model.eval()

def generate_text(prompt, max_length=50):
    # Encode the prompt
    input_ids = tokenizer.encode(prompt, return_tensors='pt')
    
    # Generate text
    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1)
    
    # Decode the output
    return tokenizer.decode(output[0], skip_special_tokens=True)

# Example usage
if __name__ == "__main__":
    prompt = "Once upon a time"
    generated_text = generate_text(prompt, max_length=100)
    print(generated_text)


