# Import necessary classes from the Hugging Face 'transformers' library.
# This library provides state-of-the-art models for natural language processing tasks.
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# Choose a model from Hugging Face Hub (e.g., 'gpt2')
model_name = "gpt2"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name) # 
model = AutoModelForCausalLM.from_pretrained(model_name)

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Input prompt
prompt = "Once upon a time"

# Run inference: In machine learning, inference means using a trained model to make
# predictions or generate outputs from new, unseen data.
outputs = generator(
    prompt,
    max_length=50,           # Maximum length of generated text
    num_return_sequences=1   # Number of generated sequences to return
)

# Print the output
print(outputs[0]['generated_text'])