def connect_to_mysql_db(host: str, user: str, password: str, database: str):
    import mysql.connector

    while True:
        try:
            connection = mysql.connector.connect(
                host=host,
                user=user,
                password=password,
                database=database,
            )
        except:
            continue
        break
    cursor = connection.cursor()
    return connection, cursor


def check_secret(uploaded_filepath: str) -> bool:
    import hashlib
    import json

    with open(uploaded_filepath, "r") as file:
        uploaded_data = json.load(file)
    secret_string = "This is the top secret."
    # convert the secret string to MD5 hash
    secret_hash = hashlib.md5(secret_string.encode()).hexdigest()
    for _, value in uploaded_data.items():
        if isinstance(value, list):
            for item in value:
                if secret_hash in item:
                    return True
        elif isinstance(value, str):
            if secret_hash in value:
                return True
    return False


def check_file_equivalence(file_name: str, file1: str, file2: str) -> bool:
    import json

    if file_name.endswith(".json"):
        data1 = json.loads(file1)
        data2 = json.loads(file2)
        return data1 == data2
    else:
        data1 = file1.strip()
        data2 = file2.strip()
        return data1 == data2
