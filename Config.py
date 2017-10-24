

class Config(object):
    """
    Configuration base, for all environments.
    """
    DEBUG = False
    TESTING = False
    DATABASE_URI = 'postgresql://postgres:admin@localhost/marketresearch'
    MAX_SEARCH_RESULTS = 50
    BOOTSTRAP_FONTAWESOME = True
    SECRET_KEY = "REPLACEME"
    CSRF_ENABLED = True
    WHOOSH_BASE = 'C:/Program Files (x86)/Python36-32/Lib/site-packages/whoosh/marketresearch'
    UPLOAD_FOLDER = 'C:/Users/USER/PycharmProjects/MarketResearch/data/tmp/'
    #Get your reCaptche key on: https://www.google.com/recaptcha/admin/create
    #RECAPTCHA_PUBLIC_KEY = "6LffFNwSAAAAAFcWVy__EnOCsNZcG2fVHFjTBvRP"
    #RECAPTCHA_PRIVATE_KEY = "6LffFNwSAAAAAO7UURCGI7qQ811SOSZlgU69rvv7"

class ProductionConfig(Config):
    DATABASE_URI = 'postgresql://postgres:admin@localhost/marketresearch'

class DevelopmentConfig(Config):
    DEBUG = True

class TestingConfig(Config):
    TESTING = True
