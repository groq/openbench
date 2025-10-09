import requests
import json
from grader import Grader


class FluentBitGrader(Grader):
    def check_service_health(self) -> bool:
        target_url = f"http://server:9090"
        payload = {"output": "stdout", "params": {"format": "json"}, "inputs": [50]}
        try:
            response = requests.post(
                target_url,
                data=json.dumps(payload),
                headers={"Content-Type": "application/json"},
            )
            return response.status_code == 200
        except Exception:
            return False
