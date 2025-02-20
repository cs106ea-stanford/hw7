# CS106A Exploring Artificial Intelligence
# Patrick Young, Stanford CS106EA

# ### Standard Imports
import torch
import transformers
import numpy as np
import matplotlib.pyplot as plt
import ipywidgets as widgets
from ipywidgets import HBox, VBox, HTML, Button, Output, Text
from IPython.display import clear_output
from transformers import AutoModel, AutoTokenizer
from typing import List, TypedDict

plt.ion()

### Load Model and Tokenizer
model_name = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name, output_attentions=True, attn_implementation="eager")

## Tokenizer
class TokenizerOutput(TypedDict):
    input_ids: torch.Tensor
    attention_mask: torch.Tensor
    token_type_ids: torch.Tensor
    
def tokenize(sentence: str) -> TokenizerOutput: 
    return tokenizer(sentence, return_tensors="pt")

def tokens_to_str(tokens: TokenizerOutput) -> List[str]:
    return tokenizer.convert_ids_to_tokens(tokens['input_ids'][0])

def draw_attention_grid(tokens: TokenizerOutput, attention_matrix: np.ndarray, include_special_tokens: bool = False):
    token_strings = tokens_to_str(tokens)
    if not include_special_tokens:
        token_strings = token_strings[1:-1]
        attention_matrix = attention_matrix[1:-1,1:-1]
    
    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(attention_matrix, cmap='Blues')
    cbar = ax.figure.colorbar(im, ax=ax)
    cbar.ax.set_ylabel("Attention", rotation=-90, va="bottom")
    ax.set_xticks(np.arange(len(token_strings)))
    ax.set_yticks(np.arange(len(token_strings)))
    ax.set_xticklabels(token_strings)
    ax.set_yticklabels(token_strings)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("BERT Attention Heatmap")
    plt.show()

def draw_attention_heads(tokens: TokenizerOutput, attention_tensor: torch.Tensor, include_special_tokens: bool = False):
    token_strings = tokens_to_str(tokens)
    if not include_special_tokens:
        token_strings = token_strings[1:-1]
        attention_tensor = attention_tensor[:, 1:-1, 1:-1]
    
    num_heads = attention_tensor.shape[0]
    fig, axes = plt.subplots(3, 4, figsize=(15, 10))  # Grid layout for 12 heads
    axes = axes.flatten()
    
    for i in range(num_heads):
        ax = axes[i]
        im = ax.imshow(attention_tensor[i].detach().numpy(), cmap='Blues')
        ax.set_title(f'Head {i}')
        ax.set_xticks(range(len(token_strings)))
        ax.set_yticks(range(len(token_strings)))
        ax.set_xticklabels(token_strings, rotation=45, ha="right")
        ax.set_yticklabels(token_strings)
        fig.colorbar(im, ax=ax)
    
    plt.tight_layout()
    plt.show()

attention_output = Output()
attention_sentence_label = HTML(value="<b>Enter text:</b>", layout={'width': '70px'})
attention_input_text = Text(value="The tree bark had lichen growing on it", layout={'width': '550px'})
show_special_tokens = widgets.Checkbox(value=False, description='Show Special Tokens', indent=False)
layer_selection = widgets.Dropdown(options=list(range(12)), value=0, layout={'width': '50px'})
attention_generate_button = Button(description="Generate Attention Map", button_style='info', layout={'width': '250px'})
attention_heads_button = Button(description="Show Individual Heads", button_style='info', layout={'width': '250px'})

stored_attention_matrices = {}
stored_tokens = None
showing_individual_heads = False

def generate_attention(_):
    global stored_attention_matrices, stored_tokens, showing_individual_heads
    with attention_output:
        clear_output(wait=True)
        input_text = attention_input_text.value
        include_special_tokens = show_special_tokens.value
        selected_layer = layer_selection.value
        stored_tokens = tokenize(input_text)
        outputs = model(**stored_tokens)
        stored_attention_matrices = {layer: outputs.attentions[layer].mean(dim=1).squeeze().detach().numpy() for layer in range(len(outputs.attentions))}
        draw_attention_grid(stored_tokens, stored_attention_matrices[selected_layer], include_special_tokens)
        showing_individual_heads = False
        attention_heads_button.description = "Show Individual Heads"

def toggle_attention_maps(_):
    global showing_individual_heads
    with attention_output:
        clear_output(wait=True)
        input_text = attention_input_text.value
        include_special_tokens = show_special_tokens.value
        selected_layer = layer_selection.value
        tokens = stored_tokens
        outputs = model(**tokens)
        
        if showing_individual_heads:
            attention_matrix = outputs.attentions[selected_layer].mean(dim=1).squeeze().detach().numpy()
            draw_attention_grid(tokens, attention_matrix, include_special_tokens)
            attention_heads_button.description = "Show Individual Heads"
        else:
            attention_tensor = outputs.attentions[selected_layer].squeeze()
            draw_attention_heads(tokens, attention_tensor, include_special_tokens)
            attention_heads_button.description = "Show Combined Heat Map"
        
        showing_individual_heads = not showing_individual_heads

def update_attention_on_layer_change(change):
    if stored_tokens is None or not stored_attention_matrices:
        return
    with attention_output:
        clear_output(wait=True)
        include_special_tokens = show_special_tokens.value
        selected_layer = change['new']
        
        if showing_individual_heads:
            attention_tensor = model(**stored_tokens).attentions[selected_layer].squeeze()
            draw_attention_heads(stored_tokens, attention_tensor, include_special_tokens)
        else:
            draw_attention_grid(stored_tokens, stored_attention_matrices[selected_layer], include_special_tokens)

layer_selection.observe(update_attention_on_layer_change, names='value')
attention_generate_button.on_click(generate_attention)
attention_heads_button.on_click(toggle_attention_maps)

def display_attention():
    display(VBox([
        HBox([attention_sentence_label, attention_input_text, show_special_tokens]),
        HBox([HTML("<b>Layer:</b>", layout={'width': '70px'}), layer_selection, attention_generate_button, attention_heads_button]),
        attention_output
    ]))
