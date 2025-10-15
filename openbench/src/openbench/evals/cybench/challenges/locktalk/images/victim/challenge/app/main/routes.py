from flask import render_template

from . import main_blueprint

@main_blueprint.route('/', methods=['GET'])
def index():
    return render_template('/index.html')
