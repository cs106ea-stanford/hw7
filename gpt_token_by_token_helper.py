# CS106A Exploring Artificial Intelligence
# Patrick Young, Stanford CS106EA

import torch
import ipywidgets as widgets
from IPython.display import display
from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Load GPT-2 model and tokenizer
model = GPT2LMHeadModel.from_pretrained("gpt2")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token  # GPT-2 has no pad token

# Define UI components with adjusted alignment
text_input = widgets.Textarea(
    value="The quick brown fox", 
    description="Input:", 
    layout=widgets.Layout(width="80%", height="100px"), 
    style={'description_width': 'initial'}  # Ensures label is flush-left
)

generate_button = widgets.Button(description="Generate Options",button_style='info', layout=widgets.Layout(margin="10px 0px"))
choices_output = widgets.Output()

# Set list box to show all 8 options and align the label
continuation_listbox = widgets.Select(
    description="Choose:", 
    layout=widgets.Layout(width="220px", height="140px"), 
    style={'description_width': 'initial'}  # Ensures label is flush-left
)
confirm_button = widgets.Button(description="Add Token",button_style='info', layout=widgets.Layout(margin="10px 0px"))

# Container for left-aligned controls
ui_container = widgets.VBox(
    [text_input, generate_button, choices_output, confirm_button],
    layout=widgets.Layout(display="flex", flex_flow="column", align_items="flex-start")  # Left-align everything
)

# Function to get top-k token choices (fixed at 8)
def get_top_k_predictions(text, k=8):
    inputs = tokenizer.encode(text, return_tensors="pt")
    outputs = model(inputs)
    logits = outputs.logits
    last_logits = logits[:, -1, :]
    probs = torch.softmax(last_logits, dim=-1)
    top_k_probs, top_k_indices = torch.topk(probs, k)

    top_k_tokens = [tokenizer.convert_ids_to_tokens([idx])[0] for idx in top_k_indices[0]]

    formatted_results = []
    
    for token, prob in zip(top_k_tokens, top_k_probs[0].tolist()):
        # Keep "Ġ" visible but make it more readable
        if token.startswith("Ġ"):
            token_clean = f"[SPACE] {token[1:]}"  # Show it as a word with space instead of just [SPACE]
        else:
            token_clean = token  # Keep normal tokens as-is

        appended_text = text + (token.replace("Ġ", " ") if "Ġ" in token else token)
        formatted_results.append((token_clean, prob, appended_text))
        
    return formatted_results


# Function to generate choices
def on_generate_clicked(_):
    choices_output.clear_output()  # Clears previous widgets before updating
    text = text_input.value.strip()
    results = get_top_k_predictions(text, k=8)  # Fixed at 8 options

    # Create formatted choices list
    formatted_choices = [f"{token} ({prob*100:.2f}%)" for token, prob, _ in results]
    
    # Update ListBox options (ensuring no duplicates)
    continuation_listbox.options = [(formatted_choices[i], results[i][2]) for i in range(len(results))]

    with choices_output:
        display(continuation_listbox)

# Function to add selected token
def on_confirm_clicked(_):
    if continuation_listbox.value:
        text_input.value = continuation_listbox.value

# Attach button events
generate_button.on_click(on_generate_clicked)
confirm_button.on_click(on_confirm_clicked)

# Display widgets inside a left-aligned container
def display_token_by_token():
    display(ui_container)
