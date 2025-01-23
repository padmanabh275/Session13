import os
import torch
import gradio as gr
from transformers import AutoTokenizer
from config import SmolLM2Config
from model import SmolLM2Lightning

def load_model(checkpoint_path):
    """Load the trained model from checkpoint"""
    try:
        config = SmolLM2Config("config.yaml")
        model = SmolLM2Lightning.load_from_checkpoint(checkpoint_path, config=config)
        model.eval()
        
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"Model loaded on GPU: {torch.cuda.get_device_name()}")
        else:
            print("Model loaded on CPU")
            
        return model
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return None

def generate_text(prompt, max_length=100, temperature=0.7, top_p=0.9, top_k=50):
    """Generate text from prompt"""
    try:
        if model is None:
            return "Model not loaded. Please check if checkpoint exists."
            
        inputs = model.tokenizer(
            prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=model.config.model.max_position_embeddings,
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                do_sample=True,
                pad_token_id=model.tokenizer.pad_token_id,
                bos_token_id=model.tokenizer.bos_token_id,
                eos_token_id=model.tokenizer.eos_token_id
            )
        
        return model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
    except Exception as e:
        return f"Error generating text: {str(e)}"

# Load the model
print("Loading model...")
checkpoint_path = "checkpoints/smol-lm2-final.ckpt"
if not os.path.exists(checkpoint_path):
    print(f"Warning: Checkpoint not found at {checkpoint_path}")
    print("Please train the model first or specify correct checkpoint path")
    model = None
else:
    model = load_model(checkpoint_path)

# Create Gradio interface
demo = gr.Interface(
    fn=generate_text,
    inputs=[
        gr.Textbox(label="Prompt", placeholder="Enter your prompt here..."),
        gr.Slider(minimum=10, maximum=200, value=100, step=1, label="Max Length"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.7, step=0.1, label="Temperature"),
        gr.Slider(minimum=0.1, maximum=1.0, value=0.9, step=0.1, label="Top-p"),
        gr.Slider(minimum=1, maximum=100, value=50, step=1, label="Top-k")
    ],
    outputs=gr.Textbox(label="Generated Text"),
    title="SmolLM2 Text Generation",
    description="Enter a prompt and adjust generation parameters to create text with SmolLM2",
    examples=[
        ["Explain what machine learning is:", 100, 0.7, 0.9, 50],
        ["Once upon a time", 150, 0.8, 0.9, 40],
        ["The best way to learn programming is", 120, 0.7, 0.9, 50]
    ]
)

def find_free_port(start_port=7860, max_port=7960):
    """Find a free port in the given range"""
    import socket
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('', port))
                return port
        except OSError:
            continue
    raise OSError(f"No free ports found in range {start_port}-{max_port}")

if __name__ == "__main__":
    print("Starting Gradio interface...")
    # Find a free port
    try:
        port = find_free_port()
        print(f"Using port: {port}")
        demo.launch(
            server_port=port,
            share=True
        )
    except Exception as e:
        print(f"Failed to start server: {str(e)}") 