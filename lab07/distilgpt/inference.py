import requests
import json

inference_url = "http://localhost:8080/v2/models/transformer/infer"

inference_request = {
    "inputs": [
        {
            "name": "args",
            "shape": [1],
            "datatype": "BYTES",
            "data": ["this is a test"],
        }
    ]
}

response = requests.post(inference_url, json=inference_request).json()

txt = json.loads(response['outputs'][0]['data'][0])['generated_text']
print(txt)


