import torch
from huggingface_hub import login
import os

import gradio as gr
from transformers import AutoModelForCausalLM, AutoTokenizer

# os.environ["HUGGINGFACEHUB_API_TOKEN"] = "SECRET_API_KEY"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "meta-llama/Llama-3.2-3B-Instruct"

try:
    # Käytä token-parametria
    hf_token = os.environ.get("HUGGINGFACEHUB_API_TOKEN", None)

    # Lataa tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        token=hf_token,
        padding_side="left",
        truncation_side="left",
    )

    # Lataa malli
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        token=hf_token,
        device_map="auto",
        torch_dtype=torch.float16, 
        low_cpu_mem_usage=True,
    )

    print("Model and tokenizer loaded successfully.")
except Exception as e:
    print(f"Error loading model or tokenizer: {e}")

SYSTEM_PROMPT = """You are a helpful assistant that always responds helpfully and informatively."""

def generate_response(prompt, history=[]):
    # Rakennetaan kokonaispromptti: ensin system prompt, sitten mahdollinen keskusteluhistoria ja lopuksi käyttäjän viesti.
    conversation = SYSTEM_PROMPT + "\n"
    for user_input, bot_response in history:
        conversation += f"User: {user_input}\nAssistant: {bot_response}\n"
    conversation += f"User: {prompt}\nAssistant:"

    inputs = tokenizer(conversation, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=300,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id,
        )

    output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Poimitaan mallin vastaus conversation-stringistä
    bot_response = output_text[len(conversation):].strip().split("User:")[0].strip()

    history.append((prompt, bot_response))
    return bot_response, history

with gr.Blocks() as demo:
    chatbot = gr.Chatbot()
    state = gr.State([])  # Initialize conversation history

    with gr.Row():
        txt = gr.Textbox(
            show_label=False,
            placeholder="Type your message and press Enter",
            lines=1,
            container=False
        )

    def respond(user_input, history):
        bot_response, history = generate_response(user_input, history)
        return "", history, history

    txt.submit(respond, [txt, state], [txt, chatbot, state])

    def clear_conversation():
        return [], []

    with gr.Row():
        clear_btn = gr.Button("Clear Conversation")
        clear_btn.click(clear_conversation, inputs=None, outputs=[chatbot, state])

demo.launch(share=True)
