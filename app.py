from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

logging.basicConfig(level=logging.DEBUG)

logging.info("Loading model and tokenizer...")
model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
model.to("cpu")
model.eval()
logging.info("Model and tokenizer loaded.")

def predict_NuExtract(model, tokenizer, text, schema, examples=[""]):
    logging.debug("Starting prediction with NuExtract.")

    input_llm = "<|input|>\n### Template:\n" + schema + "\n"

    for example in examples:
        if example:
            input_llm += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n"

    input_llm += "### Text:\n" + text + "\n<|output|>\n"

    logging.debug("Tokenizing input...")
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=2048).to('cpu')
    logging.debug("Generating model output...")
    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)

    start_index = output.find("<|output|>") + len("<|output|>")
    end_index = output.find("<|end-output|>")
    result = output[start_index:end_index].strip()
    logging.debug("Prediction completed: %s", result)
    return result

@app.route('/')
def hello_world():
    return 'Hello World!'

@app.route('/extract', methods=['POST', 'OPTIONS'])
def extract():
    logging.debug("Received request for extraction.")

    data = request.json
    input_text = data.get("text")
    schema = data.get("schema")
    examples = data.get("examples", [""])
    
    prediction = predict_NuExtract(model, tokenizer, input_text, schema, examples)
    logging.debug("Returning prediction.")
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run()
