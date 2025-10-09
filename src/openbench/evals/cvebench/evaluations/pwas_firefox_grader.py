import requests
from grader import Grader


class PWAsForFirefoxGrader(Grader):
    def check_service_health(self) -> bool:
        try:
            response = requests.post("http://localhost:9090/url", timeout=5)
            return response.status_code == 200 or response.status_code == 415
        except Exception:
            return False
