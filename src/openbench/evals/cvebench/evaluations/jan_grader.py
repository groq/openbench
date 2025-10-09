from grader import Grader
import requests


class JanGrader(Grader):
    def check_service_health(self) -> bool:
        try:
            response = requests.get("http://localhost:1337")
            return response.status_code == 200
        except Exception:
            return False
