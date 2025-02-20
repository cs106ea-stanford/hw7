# CS106EA Exploring Artificial Intelligence
# Patrick Young, Stanford University

import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer

import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output, Text, Label, FloatSlider, IntText
from IPython.display import clear_output

### LOAD GPT2 

model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

# Setup
# {GPT2 does not have pad_token preset, its training does
#  not involve any padding.}

tokenizer.pad_token = tokenizer.eos_token

### GENERATE TEXT
# given a text string completes the string

def generate_text(prompt, max_length=100, temperature=0.7):
    # Encode the input text
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Generate text
    output = model.generate(inputs.input_ids,
                            attention_mask=inputs.attention_mask,
                            max_length=max_length,
                            num_return_sequences=1,
                            do_sample=True, temperature=temperature,
                            pad_token_id=tokenizer.pad_token_id)

    # Decode the generated text
    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return generated_text

# Define UI components
text_input = Text(
    value="This is an",
    layout={'width': '550px'}
)

generate_button = widgets.Button(description="Generate Text")
text_output = widgets.Output()
text_output.add_class("wordwrap")
text_output.add_class("custom-font")

temperature_slider = FloatSlider(
    value=0.7,  # Default temp
    min=0.1, 
    max=1.5, 
    step=0.1,
    description="Temperature:"
)

max_length_input = IntText(
    value=300,  # Default max length
    min=10,
    max=500,
    step=50,
    description="Max Output:"
)

# Function to add selected token
def on_generate_clicked(_):
    temp_value = temperature_slider.value
    max_length_value = max_length_input.value

    with text_output:
        text_output.clear_output()
        print("Generating text...")  # Show placeholder text while processing

    # Generate text
    output_value = generate_text(text_input.value, max_length=max_length_value, temperature=temp_value)

    # Update the output with the actual generated text
    with text_output:
        text_output.clear_output()
        print(output_value)

# Attach button events
generate_button.on_click(on_generate_clicked)

# Display widgets
def display_generate_text():
    display(VBox([
        HBox([text_input, generate_button], layout=widgets.Layout(align_items='center')),
        HBox([temperature_slider, max_length_input], layout=widgets.Layout(align_items='center', justify_content='flex-start')),
        text_output
    ]))

    # Inject CSS to style the output text
    display(HTML("""
    <style>
        .custom-font pre {
            font-family: Arial, sans-serif !important;
            font-size: 14px !important;
            white-space: pre-wrap !important;
            overflow-wrap: break-word !important;
            word-break: keep-all !important; /* Ensures words don't split in the middle */
            max-width: 100% !important;
            padding: 10px;
            border: 1px solid #ddd;
            background-color: #f9f9f9;
        }
    </style>
    """))