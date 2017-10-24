from flask_sqlalchemy import SQLAlchemy
#import sqlalchemy as sql

db=SQLAlchemy()
class Performance(db.Model):
    __tablename__ = "perfomance"
   # metadata = sql.MetaData()
    id_performance = db.Column(db.Integer, primary_key=True)
    tp = db.Column(db.Integer)
    tn = db.Column(db.Integer)
    fp = db.Column(db.Integer)
    fn = db.Column(db.Integer)

