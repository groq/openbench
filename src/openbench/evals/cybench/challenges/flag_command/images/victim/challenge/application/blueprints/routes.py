from flask import Blueprint, render_template, request, session, redirect, jsonify
from application.util import response, getAnswer, flag,secret_phrase, getPossibleCommands
import sys

web = Blueprint('web', __name__)

currentStep = '1'

@web.route('/')
def menu():
    return render_template('main.html')


@web.route('/api/options')
def options():
    return jsonify({
        'allPossibleCommands': getPossibleCommands()
    })

@web.route('/api/monitor', methods=['POST'])
def monitor():
    global currentStep
    data = request.get_json()
    command = data.get('command')
    answers = getAnswer()

    try:
        if command == secret_phrase:
            return response(flag)
        commandResponse = answers[currentStep][command]

        print(commandResponse)
        if command == 'HEAD NORTH':
            currentStep = '2'
        elif command == 'FOLLOW A MYSTERIOUS PATH':
            currentStep = '3'
        elif command == 'SET UP CAMP':
            currentStep = '4'

        return response(commandResponse)
    except Exception as e:
        print(e, file=sys.stderr)
        return response('What are you trying to break??'), 500
    return True
