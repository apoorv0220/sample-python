# import os
# import http.server
# import socketserver

# from http import HTTPStatus


# class Handler(http.server.SimpleHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(HTTPStatus.OK)
#         self.end_headers()
#         msg = 'Hello! you requested %s' % (self.path)
#         self.wfile.write(msg.encode())


# port = int(os.getenv('PORT', 80))
# print('Listening on port %s' % (port))
# httpd = socketserver.TCPServer(('', port), Handler)
# httpd.serve_forever()

import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

app = Flask(__name__)
CORS(app)

logging.info("Loading model and tokenizer.")
try:
    model = AutoModelForCausalLM.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained("numind/NuExtract-1.5-tiny", trust_remote_code=True)
    model.to("cpu")
    model.eval()
    logging.info("Model and tokenizer loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model and tokenizer: {e}")

@app.route('/', methods=['GET'])
def index():
    logging.debug("Received request for index route.")
    msg = f"Hello! you requested {request.path}"
    return msg

def predict_NuExtract(model, tokenizer, text, schema, examples=[""]):
    logging.debug(f"Input text: {text}")
    logging.debug(f"Schema: {schema}")
    logging.debug(f"Examples: {examples}")

    input_llm = "<|input|>\n### Template:\n" + schema + "\n"
    for example in examples:
        if example:
            logging.debug(f"Processing example: {example}")
            input_llm += "### Example:\n" + json.dumps(json.loads(example), indent=4) + "\n" 
    input_llm += "### Text:\n" + text + "\n<|output|>\n"
    
    logging.debug("Tokenizing input data.")
    try:
        input_ids = tokenizer(input_llm, return_tensors="pt", truncation=True, max_length=2048).to('cpu')
    except Exception as e:
        logging.error(f"Error during tokenization: {e}")
        return "Error during tokenization"
    
    logging.debug("Generating output with model.")
    try:
        output = tokenizer.decode(model.generate(**input_ids)[0], skip_special_tokens=True)
    except Exception as e:
        logging.error(f"Error during model generation: {e}")
        return "Error during model generation"
    
    start_index = output.find("<|output|>") + len("<|output|>")
    end_index = output.find("<|end-output|>")
    prediction = output[start_index:end_index].strip()
    logging.debug(f"Model output: {prediction}")

    return prediction

@app.route('/extract', methods=['POST'])
def extract():
    logging.debug("Received request for extract route.")
    try:
        data = request.json
        input_text = data.get("text")
        schema = data.get("schema")
        examples = data.get("examples", [""])

        logging.debug(f"Input text: {input_text}")
        logging.debug(f"Schema: {schema}")
        logging.debug(f"Examples: {examples}")

        prediction = predict_NuExtract(model, tokenizer, input_text, schema, examples)
        return jsonify({"prediction": prediction})

    except Exception as e:
        logging.error(f"Error processing extract route: {e}")
        return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    logging.info(f"Starting Flask app on port {port}.")
    app.run(host='0.0.0.0', port=port)
