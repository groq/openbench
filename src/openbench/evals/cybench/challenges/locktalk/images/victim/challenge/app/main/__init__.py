from flask import Blueprint

main_blueprint = Blueprint('main_blueprint', __name__, template_folder='templates', static_folder='static')

from . import routes
