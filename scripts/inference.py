from llamafactory.chat import ChatModel
from llamafactory.extras.misc import torch_gc

# Model Configuration
args = dict(
    model_name_or_path="/home/Allwin/Fine_tune/model_finetune_arabic",
    template="llama3",
    finetuning_type="lora",
    quantization_bit=4,
)
chat_model = ChatModel(args)

# Prompt Template
PROMPT_TEMPLATE = """{instruction}

Input: {input}

Output: """

# Function to format prompt
def format_prompt(user_input):
    return PROMPT_TEMPLATE.format(instruction="Translate to Arabic:", input=user_input)

# Chat loop
messages = []
print("Welcome to the CLI application, use `clear` to remove history, use `exit` to exit.")

while True:
    query = input("\nUser: ")
    if query.strip().lower() == "exit":
        break
    if query.strip().lower() == "clear":
        messages = []
        torch_gc()
        print("History has been removed.")
        continue

    formatted_query = format_prompt(query)
    messages.append({"role": "user", "content": formatted_query})
    
    print("Assistant: ", end="", flush=True)
    response = ""
    
    for new_text in chat_model.stream_chat(messages):
        print(new_text, end="", flush=True)
        response += new_text
    
    print()
    # messages.append({"role": "assistant", "content": response})

torch_gc()
