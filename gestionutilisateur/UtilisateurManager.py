import os
from functools import wraps


from flask import flash
from flask import redirect
from flask import render_template
from flask import request
from flask import url_for
from sqlalchemy import engine
from flask import session
from sqlalchemy.orm import sessionmaker
from flask.blueprints import Blueprint
from gestionutilisateur.UtilisateurDB import Utilisateur, db
from werkzeug.security import generate_password_hash, check_password_hash


user = Blueprint('user', __name__, template_folder='templates', url_prefix='', static_folder='static')
user.secret_key = os.urandom(12)



def requires_roles(*roles):
    def wrapper(f):
        @wraps(f)
        def wrapped(*args, **kwargs):
            if get_current_user_role() not in roles:
                flash("vous n'avez pas le droit d'acceder a cette page ", 'error')
                return redirect(url_for('user.login'))
            return f(*args, **kwargs)
        return wrapped
    return wrapper

def login_required(f):
        @wraps(f)
        def wrap(*args, **kwargs):
            if 'logged_in' in session:
                flash('heyyyyyyyy', 'error')
                return f(*args, **kwargs)
            else:
                flash("you need to login first")
                return redirect(url_for('user.login'))

        return wrap



def get_current_user_role():
    if not session.get('role'):
        return 'null'
    else:
        return session['role']

@user.route('/inscription', methods=['GET', 'POST'])
#@login_required
#@requires_roles('admin')
def inscription():
   if request.method == "GET":
        return render_template('inscription.html')
   if request.method == "POST":
       required = ['login', 'password', 'nom', 'prenom', 'email']
       for r in required:
           if r not in request.form:
               flash("Error: {0} is required.".format(r))
               return redirect(url_for('inscription'))
       utlisateur = Utilisateur(login=request.form['login'], password=request.form['password'], nomUtilisateur=request.form['nom'],
                                prenomUtilisateur=request.form['prenom'], email=request.form['email'],
                                poste=request.form['poste'],active=True,role='user')
       db.session.add(utlisateur)
       db.session.commit()
       flash('User successfully registered')
       return afficherUtilisateurs()





@user.route('/login', methods=['GET','POST'])
def login():
    if request.method == "GET":

        return render_template('login.html')
    if request.method == "POST":
        postLogin = str(request.form['login'])
        postPassword = str(request.form['password'])
        role='user'
        Session = sessionmaker(bind=engine)
        s = Session()
        result = Utilisateur.query.filter_by(login=postLogin).first()
        print('result',result)
        if result and check_password_hash(result.password, postPassword):

            session['logged_in'] = True
            session['role']=result.role
            print('role',role,'    ',result.role)
            role=result.role
        else:
            flash('wrong password!')
        if (role=='admin'):
            return  render_template('home.html')
        else:
            return render_template('home2.html')


@user.route('/')
@login_required
def home():
    if not session.get('logged_in'):
        return render_template('login.html')
    else:
        return "Hello Boss!  <a href='/logout'>Logout</a>"

@user.route('/afficherUtilisateurs')
#@login_required
def afficherUtilisateurs():
    users=Utilisateur.query.all()
    return render_template('afficherUtilisateurs.html', users=users)


@user.route('/supprimerUtilisateur/<int:id>', methods=['POST'])
#@login_required
def supprimerUtilisateur(id):
    Utilisateur.query.filter_by(id_utilisateur=id).delete()
    db.session.commit()
    return redirect(url_for('user.afficherUtilisateurs'))


@user.route('/modifierUtilisateur/<int:id>', methods=['GET','POST'])
#@login_required
def modifierUtilisateur(id):
    utilisateur= Utilisateur.query.get(id)
    return render_template('modifierUtilisateur.html', utilisateur= utilisateur,id=id)

@user.route('/postmodifierUtilisateur/<int:id>', methods=[ 'POST'])
#@login_required
def postmodifierUtilisateur(id):

    utilisateur=Utilisateur.query.filter_by(id_utilisateur=id).update(dict(login=request.form['login'],
                                password=request.form['password'], nomUtilisateur=request.form['nom'],
                                prenomUtilisateur=request.form['prenom'], email=request.form['email'],
                                poste=request.form['poste']))
    db.session.commit()
    return redirect(url_for('user.afficherUtilisateurs'))

@user.route("/logout")
@login_required
def logout():
    session['logged_in'] = False
    return home()


