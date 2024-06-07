import argparse
from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F

app = Flask(__name__)

# Initialize the tokenizer and model
tokenizer = None
model = None


def initialize_model(model_name):
    global tokenizer, model, eos_token_id
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    eos_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(model_name)


@app.route("/generate", methods=["POST"])
def generate():
    # Get the prompt from the request
    data = request.get_json()
    prompt = data.get("prompt", "")

    # Encode the prompt and generate a response
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    # Get the model output
    with torch.no_grad():
        output = model(inputs.input_ids, return_dict=True)

    logits = output["logits"]
    probs = F.softmax(logits, dim=-1)
    next_token_probs = probs[:, -1, eos_token_id]
    return jsonify({"probabilities": next_token_probs.tolist()})


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Flask server with transformer model."
    )
    parser.add_argument(
        "--model",
        type=str,
        default="distilgpt2",
        help="Name of the model to use.",
    )
    args = parser.parse_args()

    initialize_model(args.model)

    app.run(debug=True)
