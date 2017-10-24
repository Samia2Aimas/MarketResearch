from __future__ import unicode_literals
from modulefinder import Module
from unicodedata import name

from flask import flash
from flask import  render_template, redirect, url_for, request
from flask.blueprints import Blueprint
from sqlalchemy.orm import session
from wtforms import Form, StringField, IntegerField, TextField, SubmitField

from wtforms import form

import flask_whooshalchemy
from wtforms import validators

from Etude.etudeDb import db,Etude
blueprint = Blueprint('etude_blueprint', __name__, template_folder='templates', url_prefix='/etude', static_folder='static')

class EtudeForm(Form):
    nomEtude=StringField('nomEtude',validators=[validators.required()])
    description = TextField('description')
    duree=IntegerField('duree',[validators.length(min=30, max =365)])
    submit=SubmitField('ajouter etude')
@blueprint.route('/createEtude')
def createEtude():
 form=EtudeForm()
 return render_template("etude.html",form=form)

@blueprint.route('/index')
def index():
    myEtude = Etude.query.all()

    OneItem = Etude.query.filter_by(id=1).first()
    return render_template("index.html", myEtude=myEtude)

@blueprint.route('/creerEtude')
def creerEtude():
    return render_template("creerEtude.html")

@blueprint.route('/postEtude',methods=['POST'])
def post_etude():

     etude = Etude(nomEtude=request.form['nomEtude'], type=request.form['type'], duree=request.form['duree'],
                  periode=request.form['periode'])
     db.session.add(etude)
     db.session.commit()

     return redirect(url_for('etude_blueprint.index'))

@blueprint.route('/editEtude/<int:id>')
def editEtude(id):
    etudeExi = Etude.query.get(id)
    return render_template("modifierEtude.html", etudeExi=etudeExi ,id=id)

@blueprint.route('/supprimerEtude/<int:id>', methods=['POST'])
def supprimerEtude (id):

     etude = Etude.query.get_or_404(id)

     db.session.delete(etude)
     db.session.commit()
     return redirect(url_for('etude_blueprint.index'))

@blueprint.route('/modifierEtude/<int:id>',methods=['POST'])

def modifierEtude (id):

     etudeExi=Etude.query.filter_by(id=id).update(dict(nomEtude=request.form['nomEtude'],
                                                       type=request.form['type'],
                                                      periode=request.form['periode']))
     db.session.commit()

     return redirect(url_for('etude_blueprint.index'))



@blueprint.route('/resultat_recherche/<nom_etude>')
def resultat_recherche(nom_etude):
    print(nom_etude)

    results=Etude.whoosh_search('moii')

    return render_template('search_results.html',
                           nom_etude=nom_etude,
                           results=results)





@blueprint.route('/rechercherEtude/',methods=['POST'])
def rechercherEtude () :
   nom_etude = request.form['nomEtude']


   return redirect(url_for('etude_blueprint.resultat_recherche',nom_etude=nom_etude))


