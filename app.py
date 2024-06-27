import streamlit as st
from transformers import AutoTokenizer, AutoModelForCausalLM

# Use GPT-2 model
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Streamlit app
st.title("Blog Post Generator")
st.write("Enter a prompt to generate a blog post:")

# Text input
prompt = st.text_area("Prompt", height=150)

# Generate button
if st.button("Generate Blog Post"):
    if prompt:
        # Tokenize the input prompt
        inputs = tokenizer.encode(prompt, return_tensors="pt")

        # Generate text
        outputs = model.generate(inputs, max_length=500, num_return_sequences=1, pad_token_id=tokenizer.eos_token_id)

        # Decode the generated text
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

        st.write("## Generated Blog Post")
        st.write(generated_text)
    else:
        st.warning("Please enter a prompt.")
