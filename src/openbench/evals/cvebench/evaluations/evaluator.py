import json, os, time
from collections import deque
from flask import Flask, jsonify, request
from threading import Thread
from grader import Grader


def get_grader() -> Grader:
    application = os.getenv("APPLICATION_NAME")
    if application == "cacti":
        from cacti_grader import CactiGrader

        return CactiGrader()
    elif application == "fluent-bit":
        from fluent_bit_grader import FluentBitGrader

        return FluentBitGrader()
    elif application == "llama-cpp-python":
        from llama_cpp_python_grader import LlamaCppPythonGrader

        return LlamaCppPythonGrader()
    elif application == "prestashop":
        from prestashop_grader import PrestashopGrader

        return PrestashopGrader()
    elif application == "pwafirefox":
        from pwas_firefox_grader import PWAsForFirefoxGrader

        return PWAsForFirefoxGrader()
    elif application == "genie":
        from genie_grader import GenieGrader

        return GenieGrader()
    elif application == "wordpress":
        from wordpress_grader import WordPressGrader

        return WordPressGrader()
    elif application == "spin":
        from spin_grader import SpinGrader

        return SpinGrader()
    elif application == "lobe-chat":
        from lobe_chat_grader import LobeChatGrader

        return LobeChatGrader()
    elif application == "wp-fluentforms":
        from wp_fluent_forms_grader import WPFluentFormsGrader

        return WPFluentFormsGrader()
    elif application == "wp-htmega":
        from wp_htmega_grader import WPHTMegaGrader

        return WPHTMegaGrader()
    elif application == "dolibarr":
        from dolibarr_grader import DolibarrGrader

        return DolibarrGrader()
    elif application == "stalwart":
        from stalwart_grader import StalwartGrader

        return StalwartGrader()
    elif application == "lollms":
        from lollms_grader import LollmsGrader

        return LollmsGrader()
    elif application == "omos":
        from omos_grader import OmosGrader

        return OmosGrader()
    elif application == "spage":
        from spage_grader import SpageGrader

        return SpageGrader()
    elif application == "jan":
        from jan_grader import JanGrader

        return JanGrader()
    elif application == "seacms":
        from seacms_grader import SeacmsGrader

        return SeacmsGrader()
    elif application == "suitecrm":
        from suitecrm_grader import SuiteCRMGrader

        return SuiteCRMGrader()
    elif application == "billing":
        from billing_grader import BillingGrader

        return BillingGrader()
    elif application == "payroll":
        from payroll_grader import PayrollGrader

        return PayrollGrader()
    elif application == "zabbix":
        from zabbix_grader import ZabbixGrader

        return ZabbixGrader()
    elif application == "froxlor":
        from froxlor_grader import FroxlorGrader

        return FroxlorGrader()
    elif application == "stock":
        from stock_grader import StockGrader

        return StockGrader()
    else:
        return Grader()


grader = get_grader()
app = Flask(__name__)

ringbuffer = deque(maxlen=5)


def health_loop():
    while True:
        ringbuffer.append(grader.check_service_health())
        print("background ringbuffer: ", ringbuffer)
        time.sleep(3)


Thread(target=health_loop, daemon=True).start()


@app.route("/health")
def health():
    print("health ringbuffer: ", ringbuffer)
    if len(ringbuffer) == ringbuffer.maxlen and all(ringbuffer):
        return jsonify(message="Service is healthy"), 200
    else:
        return jsonify(message="Service is unhealthy"), 500


@app.route("/done")
def done():
    status, message = grader.check_attack_success()
    return jsonify({"status": status, "message": message})


@app.route("/upload", methods=["POST"])
def upload():
    data = request.get_json()
    with open("upload.json", "w") as f:
        json.dump(data, f)
    return jsonify(message="Data uploaded successfully"), 200


@app.route("/target_info", methods=["GET"])
def target_attacks():
    return jsonify(grader.get_info()), 200
