import os
from flask import Flask, request, jsonify
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

PREDICTION_MODEL = os.getenv('PREDICTION_MODEL', 'facebook/bart-large-mnli')
GENERATIVE_MODEL = os.getenv('REASONING_MODEL', 'google/flan-t5-base')

# Load a Generative AI model to provide the reason
model = AutoModelForSeq2SeqLM.from_pretrained(GENERATIVE_MODEL)
tokenizer = AutoTokenizer.from_pretrained(GENERATIVE_MODEL)
model.eval()

# Load a prediction model to determine a rating and confidence score.
classifier = pipeline("zero-shot-classification", model=PREDICTION_MODEL)


def validate_claim(claim: str) -> dict:
    """
    Validates the claim that was passed given the string with the prediction model.
    :param claim: Statement to verify.
    :return: dict containing the "rating", "confidence" and "reason"
    """
    labels = ["true", "false"]

    pre_process_context = (f"Claim: '{claim}, is Is this claim true or false? "
                           f"State your estimate. Provide a clear factual reason.")
    context = generate_reason(pre_process_context)
    result = classifier(f"Context: {context} \n Claim: {claim}", candidate_labels=labels)
    prediction = result['labels'][0]
    confidence = int(round(float(result['scores'][0]) * 100))

    if confidence < 50:
        return {
            "rating": "unverified",
            "confidence": confidence,
            "reason": "The model used is unsure about this claim."
        }

    return {
        "rating": prediction,
        "confidence": confidence,
        "reason": context
    }


def generate_reason(prompt: str) -> str:
    """
    Used by the generative AI model (given the prompt) to determine the reason for the claim.
    :param prompt: The prompt to use in the AI model
    :return: String with the reason
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt", truncation=True, max_length=512)
    outputs = model.generate(inputs, max_length=150)
    reason = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if not reason or "is" not in reason:
        reason = "Unable to generate a detailed explanation at this time."
    return reason


@app.route("/truth-evaluator-service/evaluate", methods=["POST"])
def evaluate():
    """
    POST request to validate claim
    :return: HTTP response
        - 200 - confidence score, reason, and rating returned
        - 400 Invalid claim attribute
        - 500 Exception error
    """
    try:
        body_params = request.get_json()
        if body_params.get('claim') is None:
            return jsonify({"reason": "Invalid parameter ‘claim’ is given or missing."}), 400
        else:
            result = validate_claim(body_params['claim'])
        return jsonify(result), 200
    except Exception:
        return jsonify({"reason": "Service not available"}), 500


if __name__ == '__main__':
    app.run(host=os.getenv('HOST', 'localhost'), port=os.getenv('PORT', 8000))
