from flask import Flask, request, jsonify
import torch
from typing import Dict, List, Any
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# get dtype
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] == 8 else torch.float16

app = Flask(__name__)

class EndpointHandler:
    def __init__(self, path=""):
        # load the model
        self.tokenizer = AutoTokenizer.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF")
        model = AutoModelForCausalLM.from_pretrained("TheBloke/Wizard-Vicuna-13B-Uncensored-HF", device_map="auto", load_in_8bit=True)
        # create inference pipeline
        self.pipeline = pipeline("text-generation", model=model, tokenizer=self.tokenizer)

    def __call__(self, data: Any) -> Dict[str, str]:
        input = data.pop("inputs", data)

        generated_text = self.pipeline(input, max_length=len(self.tokenizer(input)["input_ids"])+250, num_return_sequences=1, bad_words_ids = [[1, 584]], temperature = 1, repetition_penalty = 1.3, batch_size=1)
        return {'generated_text': generated_text}

handler = EndpointHandler()

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No input data received'}), 400
    result = handler(data)
    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
