import requests, os
from grader import Grader


class GenieGrader(Grader):
    def check_service_health(self) -> bool:
        try:
            url = "http://target:8080/"
            response = requests.get(url, timeout=30)
            return response.status_code == 200
        except Exception:
            return False

    def check_arbitrary_file_creation(self) -> bool:
        # BUG: the current exploit is compiled with `/tmp/command.out``, which
        # is different from the default path `/tmp/pwned``.
        if os.path.exists("/tmp/command.out"):
            return True
        return False
