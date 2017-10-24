from flask.ext.sqlalchemy import SQLAlchemy

db=SQLAlchemy()
class Utilisateur(db.Model):
    __tablename__ = "utilisateur_etude"

    id= db.column(db.Integer,primary_key=True)
    id_utilisateur = db.Column(db.Integer, primary_key=True)
    id_Etude = db.Column(db.String(64), unique=False)
    visibilité = db.Column(db.String(64), index=False)