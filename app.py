from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from transformers import AutoModelForCausalLM, AutoTokenizer

app = Flask(__name__)
CORS(app)

model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)

model.to("cpu")
model.eval()

def predict_NuExtract(model, tokenizer, text, schema, examples=[""]):
    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for example in examples:
        if example:
            input_llm += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n"
    input_llm += "### Text:\n" + text + "\n<|output|>\n"
    
    input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=2048).to('cpu')
    output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    
    start_index = output.find("<|output|>") + len("<|output|>")
    end_index = output.find("<|end-output|>")
    return output[start_index:end_index].strip()

@app.route('/', methods=['GET'])
def index():
    return "Hello, World! The server is up and running."

@app.route('/extract', methods=['POST'])
def extract():
    data = request.json
    input_text = data.get("text")
    schema = data.get("schema")
    examples = data.get("examples", [""])

    prediction = predict_NuExtract(model, tokenizer, input_text, schema, examples)
    return jsonify({"prediction": prediction})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
