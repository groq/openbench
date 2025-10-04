from application.util import generate

class Config(object):
    SECRET_KEY = generate(50)

class ProductionConfig(Config):
    pass

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True