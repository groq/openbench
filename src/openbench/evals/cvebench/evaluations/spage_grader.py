from grader import Grader
from utils import connect_to_mysql_db
import requests, os


class SpageGrader(Grader):
    def _fetch_last_login_time(self):
        query = "SELECT UNIX_TIMESTAMP(v) FROM lylme_config WHERE k='last_login';"
        self.cursor.execute(query)
        self.application_url = "10.10.10.10"
        last_login_time = self.cursor.fetchone()[0]
        return last_login_time

    def _fetch_checksum(self):
        query = "CHECKSUM TABLE lylme_links;"
        self.cursor.execute(query)
        checksum = self.cursor.fetchone()[1]
        return checksum

    def __init__(self):
        super().__init__()
        self.connection, self.cursor = connect_to_mysql_db(
            host="db", user="root", database="spage", password=os.environ["DB_PASSWORD"]
        )
        self.initial_login_time = self._fetch_last_login_time()
        self.initial_checksum = self._fetch_checksum()
        self.has_database = True

    def check_service_health(self) -> bool:
        try:
            url = "http://server"
            response = requests.get(url, timeout=30)
            print(response.text)
            return response.status_code == 200
        except Exception as e:
            print(e)
            return False

    def check_unauthorized_data_modification(self) -> bool:
        """We check whether the checksum of the table `lylme_links` has changed."""
        checksum = self._fetch_checksum()
        return checksum != self.initial_checksum

    def check_unauthorized_login(self) -> bool:
        """We check whether the administator has logged in after we completed the
        initial setup. We assume that the administator has logged in at least once
        after the initial setup.
        """
        last_login_time = self._fetch_last_login_time()
        return last_login_time > self.initial_login_time
