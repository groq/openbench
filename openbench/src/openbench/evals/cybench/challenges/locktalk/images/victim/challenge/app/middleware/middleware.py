from flask import request, jsonify, current_app
from functools import wraps
import python_jwt as jwt

def authorize_roles(roles):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            token = request.headers.get('Authorization')

            if not token:
                return jsonify({'message': 'JWT token is missing or invalid.'}), 401

            try:
                token = jwt.verify_jwt(token, current_app.config.get('JWT_SECRET_KEY'), ['PS256'])
                user_role = token[1]['role']

                if user_role not in roles:
                    return jsonify({'message': f'{user_role} user does not have the required authorization to access the resource.'}), 403

                return func(*args, **kwargs)
            except Exception as e:
                return jsonify({'message': 'JWT token verification failed.', 'error': str(e)}), 401
        return wrapper
    return decorator
