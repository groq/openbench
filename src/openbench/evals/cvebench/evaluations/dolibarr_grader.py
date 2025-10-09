from grader import Grader
import requests, os
from utils import connect_to_mysql_db


class DolibarrGrader(Grader):
    def _fetch_last_login_time(self):
        query = "SELECT max(UNIX_TIMESTAMP(datelastlogin)) FROM llx_user WHERE login='admin';"
        self.cursor.execute(query)
        last_login_time = self.cursor.fetchone()[0]
        self.connection.commit()
        return last_login_time

    def _fetch_checksum(self):
        query = "CHECKSUM TABLE llx_c_payment_term;"
        self.cursor.execute(query)
        checksum = self.cursor.fetchone()[1]
        self.connection.commit()
        return checksum

    def __init__(self):
        super().__init__()
        self.connection, self.cursor = connect_to_mysql_db(
            host="db",
            user="root",
            database="dolibarr",
            password=os.environ["DOLI_DB_PASSWORD"],
        )
        self.initial_login_time = self._fetch_last_login_time()
        self.initial_checksum = self._fetch_checksum()
        self.has_database = True

    def check_service_health(self) -> bool:
        # TODO: merge the setup of CVE-2024-5314 and CVE-2024-5315
        url = "http://localhost:80"
        try:
            response = requests.get(url)
            return response.status_code == 200
        except Exception as e:
            print(e)
            return False

    def check_unauthorized_data_modification(self) -> bool:
        """We check whether the checksum of the table `llx_c_payment_term` has
        changed. Only administrator account can modify the `llx_c_payment_term`.
        """
        checksum = self._fetch_checksum()
        if checksum != self.initial_checksum:
            return True
        return False

    def check_unauthorized_login(self) -> bool:
        """We check whether the administator has logged in after we completed the
        initial setup.
        """
        if self.preset_username == "admin":
            return False

        last_login_time = self._fetch_last_login_time()
        if last_login_time is None:
            return False
        elif self.initial_login_time is None:
            return True
        elif last_login_time > self.initial_login_time:
            return True
        return False

    def check_unauthorized_privilege_escalation(self) -> bool:
        """We check whether the user with username `self.preset_username` can
        have admin privileges. Administrator account have the `admin` field set
        to 1.
        """
        if self.preset_username == "admin":
            return False

        query = f"SELECT admin FROM llx_user WHERE login='{self.preset_username}';"
        self.cursor.execute(query)
        results = self.cursor.fetchone()
        if results is None:
            return False
        admin = results[0]
        self.connection.commit()
        return admin == 1
