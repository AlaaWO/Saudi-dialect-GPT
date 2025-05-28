import tkinter as tk
from tkinter import scrolledtext
import torch
from minbpe import RegexTokenizer
from transformer.model import GPTLanguageModel
import os

filepath: r'..\saudi dialect GPT demo\app'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = RegexTokenizer()
tokenizer.load(model_file=os.path.join(os.path.dirname(__file__), "tokenizer", "my_tokenizer.model"))

def get_vocab_size(tokenizer: RegexTokenizer) -> int:
    vocab = tokenizer.vocab
    special_tokens = tokenizer.special_tokens

    return len(vocab) + len(special_tokens)

#note to self: in the save checkpoint function, add configs to 
vocab_size = get_vocab_size(tokenizer)
block_size = 256
n_embd = 504
n_head = 12
n_layer = 4
dropout = 0.2
batch_size = 64

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout,
    device=device,
    ignore_index=tokenizer.special_tokens["<|padding|>"],
).to(device)
model = torch.compile(model)

checkpoint_path = model_file=os.path.join(os.path.dirname(__file__), 'finetuned', 'with context 4k lines', 'checkpoint_19.pth')
checkpoint = torch.load(checkpoint_path, weights_only=True)
model_state_dict = checkpoint["model_state_dict"]
model.load_state_dict(model_state_dict)


def get_input_tokens(message: str) -> torch.Tensor:
    input_tokens = tokenizer.encode(
        f"<|startoftext|>{message}<|separator|>", allowed_special="all")
    return torch.tensor(input_tokens, dtype=torch.long).unsqueeze(0).to(device)

def generate_response(user_message: str) -> str:
    input_tokens = get_input_tokens(user_message)
    model_answer = ""
    model.eval()
    with torch.no_grad():
        while True:
            output_tokens = model.generate(input_tokens=input_tokens, max_new_tokens=1)
            last_token = output_tokens[0, -1].item()
            if last_token == tokenizer.special_tokens["<|endoftext|>"]:
                break
            input_tokens = torch.cat((input_tokens, output_tokens[:, -1:]), dim=1)
            model_answer += tokenizer.decode([last_token])
    return model_answer

BG_COLOR = "#c2ddbd" # Light blue background
TEXT_COLOR = "#0D2306"  # White text
INPUT_COLOR = "#c2ddbd"  # Darker gray for input
BUTTON_COLOR = "#0d2f07"  # Button color
BUTTON_TEXT_COLOR = "#c2ddbd"  # Button text color

# Tkinter GUI

def send_message():
    user_input = entry.get()
    if user_input.strip():
        response = generate_response(user_input)
        chat_area.insert(tk.END, f"You: {user_input}\n", "user")
        chat_area.insert(tk.END, f"Assistant: {response}\n\n", "assistant")
        entry.delete(0, tk.END)
        chat_area.see(tk.END) 

def on_send():
    user_input = entry.get()
    if user_input.strip():
        response = generate_response(user_input)
        chat_area.insert(tk.END, f"You: {user_input}\n")
        chat_area.insert(tk.END, f"Assistant: {response}\n\n")
        entry.delete(0, tk.END)

root = tk.Tk()
root.title("Saudi Dialect GPT Demo")
root.configure(bg=BG_COLOR)
root.geometry("600x400")  # Initial size



# Configure grid weights for resizing
root.grid_rowconfigure(0, weight=1)  # Chat area will expand
root.grid_rowconfigure(1, weight=0)  # Input area stays fixed height
root.grid_columnconfigure(0, weight=1)  # Single column


chat_area = scrolledtext.ScrolledText(
    root,
    wrap=tk.WORD,
    bg=BG_COLOR,
    fg=TEXT_COLOR,
    insertbackground=TEXT_COLOR,
    font=("Helvetica", 12))
chat_area.grid(row=0, column=0, sticky="nsew", padx=10, pady=10)

WELCOME_MESSAGE = "This model was trained on limited resources and data. the base model has 14 million parameters\ndue to the limited data, the model does provide accurate or relevant responses. It was finetuned using full context finetuning"
chat_area.insert(tk.END, f"Assistant: {WELCOME_MESSAGE}\n\n", "assistant")
chat_area.see(tk.END)  

#Input Frame (Bottom Section) 
input_frame = tk.Frame(root, bg=BG_COLOR)
input_frame.grid(row=1, column=0, sticky="ew", padx=10, pady=(0,10))

# Input field (expands horizontally)
entry = tk.Entry(
    input_frame,
    width=40,
    bg=INPUT_COLOR,
    fg=TEXT_COLOR,
    font=("Helvetica", 12))
entry.pack(side=tk.LEFT, expand=True, fill=tk.X, padx=(0,5))

# Send button (fixed width)
send_button = tk.Button(
    input_frame,
    text="Send",
    command=send_message,
    bg=BUTTON_COLOR,
    fg=BUTTON_TEXT_COLOR,
    relief=tk.FLAT)
send_button.pack(side=tk.RIGHT)

# Bind Enter key
entry.bind("<Return>", lambda event: send_message())

# Make the entry focus by default
entry.focus_set()

root.mainloop()
