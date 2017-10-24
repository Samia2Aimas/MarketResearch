
from flask_sqlalchemy import SQLAlchemy
from flask_login import unicode
from werkzeug.security import generate_password_hash, check_password_hash

db=SQLAlchemy()
class Utilisateur(db.Model):
    __tablename__ = "utilisateur"

    id_utilisateur = db.Column(db.Integer, primary_key=True)
    nomUtilisateur = db.Column(db.String(64), unique=False)
    prenomUtilisateur = db.Column(db.String(64), index=False)
    login = db.Column(db.String(64), unique=True)
    email= db.Column(db.String(64),  unique=True)
    password = db.Column(db.String(120))
    active=db.Column(db.Boolean())
    role=db.Column(db.String(64))
    poste=db.Column(db.String(120))


    def set_password(self, password):
        self.password = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password, password)

    def __init__(self, login,password, nomUtilisateur,prenomUtilisateur,email,role,poste,active):
     self.login=login
     self.nomUtilisateur=nomUtilisateur
     self.prenomUtilisateur=prenomUtilisateur
     self.email=email
     self.password=generate_password_hash(password)
     self.role=role
     self.poste=poste
     self.active=active

    def __repr__(self):
        return '<Utilisateur %r>' % (self.login)

    def is_authenticated(self):
        return True

    def is_active(self):
        return True

    def is_anonymous(self):
        return False

    def get_idUtilisateur(self):
        try:
            return unicode(self.id_utilisateur)
        except NameError:
            return str(self.id)

