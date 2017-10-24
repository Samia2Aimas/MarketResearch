import pandas as pd
import pygal
from flask import Flask
from flask import Flask
from flask import Response
from flask import redirect
from flask import render_template
from flask_babel import Babel
from flask import request
from flask import url_for
from flask_appbuilder import AppBuilder
from flask_appbuilder import GroupByChartView
from flask_appbuilder import ModelView
from flask_appbuilder import aggregate_count
from flask_appbuilder.models.sqla.interface import SQLAInterface
from sqlalchemy.orm import session
from whoosh.analysis import StemmingAnalyzer
import flask_whooshalchemy as whooshalchemy
from whoosh.index import FileIndex
from wtforms import form
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash
import os
from ModelPreprocessing.ModelPreprocessingManager import model
from ModelPreprocessing.ModelPreprocessingManager2 import view

from ModelProcessing.customerSegmentation import model3
from gestionutilisateur.UtilisateurDB import Utilisateur
from gestionutilisateur.UtilisateurManager import user
from ModelProcessing.performanceDB import db




app = Flask(__name__)
app = Flask(__name__, static_folder='static')
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://postgres:admin@localhost/marketresearch'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['WHOOSH_BASE'] = 'C:/Program Files (x86)/Python36-32/Lib/site-packages/whoosh/marketresearch'
app.config['SECRET_KEY']='super-secret'
app.config['UPLOAD_FOLDER'] = 'data/'
# These are the extension that we are accepting to be uploaded
app.config['ALLOWED_EXTENSIONS'] = set(['csv'])
from Etude.etudeDb import  Etude
from whoosh.filedb.filestore import FileStorage
from whoosh.fields import Schema, TEXT, ID



#appbuilder = AppBuilder(app, db.session)
#from ModelPreprocessing.DossierTravailDb import db
from ModelPreprocessing.FichierDb import db
#from gestionutilisateur.UtilisateurDB import db, Utilisateur
from Etude.etudeDb import db
LANGUAGES ={
    "en":'English'  ,
    'es':'Espagnol'
}
db.init_app(app)
babel=Babel(app)
@app.route('/index')
def index():
    return render_template("home.html")

@app.before_first_request
def create_user():
     print('heee')
     db.create_all()
     if not Utilisateur.query.filter_by(login='imane').first():
         db.session.add(Utilisateur(login='imane',password='imane',nomUtilisateur='i',prenomUtilisateur='i',email='a',poste='a',active=True,role="admin"))
     db.session.commit()

@babel.localeselector
def get_locale():
    return request.accept_languages.best_match(LANGUAGES.keys())
from Etude.etudeManager import blueprint
from ModelPreprocessing.ModelProcessing import model2
app.register_blueprint(blueprint)
app.register_blueprint(model)
app.register_blueprint(user)
app.register_blueprint(model2)
app.register_blueprint(view)
app.register_blueprint(model3)




if __name__ == '__main__':
  app.secret_key = os.urandom(12)
  app.run(debug=True)
