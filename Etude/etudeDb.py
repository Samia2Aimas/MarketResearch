import sys
from flask_appbuilder import SQLA
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, engine
from sqlalchemy import Integer
from sqlalchemy import String

#db = SQLA()
import flask_whooshalchemy



enable_search = False
if sys.version_info >= (3, 0):
    enable_search = False
else:
    enable_search = True


db=SQLAlchemy()



class Etude(db.Model):

    __table_args__ = {'extend_existing': True}
    __searchable__ = ['nomEtude']  # these fields will be indexed by whoosh
      # configure analyzer; defaults to
                                       # StemmingAnalyzer if not specified

    id = db.Column(db.Integer, primary_key=True)
    nomEtude = db.Column(db.String(64), index=True, unique=True)
    type= db.Column(db.String(64), index=True, unique=False)
    description=db.Column(db.String(64), index=True, unique=True)
    duree=db.Column(db.Integer, index=False)
    periode=db.Column(db.DATE, index=False)

def __init__(self, nomEtude,type,description,duree,periode):
     self.nomEtude=nomEtude
     self.type=type
     self.description=description
     self.duree=duree
     self.periode=periode

def __init__(self, nomEtude):
         self.nomEtude=nomEtude


def __init__(self, nomEtude,type,periode):
    self.nomEtude = nomEtude
    self.type = type
    self.periode= periode
def __repr__(self):
        return '<Etude %r>' % (self.nomEtude)




def __repr__(self):
 return self.name


