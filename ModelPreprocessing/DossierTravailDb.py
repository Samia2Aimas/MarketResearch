from flask_sqlalchemy import SQLAlchemy

db=SQLAlchemy()
class DossierTravail(db.Model):

    id_dossierTravail = db.Column(db.Integer, primary_key=True)
    url = db.Column(db.String(120), unique=False)
    '''fichiers = db.relationship('ModelPreprocessing.FichierDb.Fichier', backref='dossier_travail',
                            lazy='dynamic')'''

    def __repr__(self):
        return '<DossierTravail:{}>'.format(self.nomDossier)

