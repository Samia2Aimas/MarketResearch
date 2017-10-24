from gettext import gettext

from flask import redirect , url_for
from flask_wtf import Form
from wtforms import TextField,validators, SelectField, StringField, IntegerField, FieldList, FormField

from Etude.etudeDb import Etude

'''
class Etudeform(Form):
    Etude.nomEtude=StringField('nomEtude')
    Etude.type = StringField('type')
    Etude.periode = IntegerField('periode')




class GroupeEtudeForm(Form):

   groupeEtude=FieldList(FormField(Etudeform)) '''

from Etude import etudeDb
from MarketResearch import db