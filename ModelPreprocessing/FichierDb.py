from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()
class Fichier(db.Model):
    __tablename__ = "fichier"

    id_fichier = db.Column(db.Integer, primary_key=True)
    nomFichier = db.Column(db.String(128), unique=False)
    taille = db.Column(db.Integer,unique=False)
    '''dossier_id = db.Column(db.Integer, db.ForeignKey('dossier_travail.id_dossierTravail'))
    dossierTravail = db.relationship('ModelPreprocessing.DossierTravailDb.DossierTravail')'''

    def __repr__(self):
        return '<Fichier:{}>'.format(self.nomFichier)
