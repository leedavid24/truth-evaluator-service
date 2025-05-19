import requests
import json

SERVICE_URL = "http://localhost:8000/truth-evaluator-service/evaluate"


def validate_a_claim():
    claim = str(input("Please enter a claim to validate: "))
    response = requests.post(SERVICE_URL, json={"claim": claim})
    if response.status_code == 200:
        response = response.json()
        print(f"The service returned a confidence score of: {response.get('confidence')}")
        print(f"The service returned a rating of: {response.get('rating')}")
        print(f"The service returned a reason: {response.get('reason')}")


while True:
    validate_a_claim()
