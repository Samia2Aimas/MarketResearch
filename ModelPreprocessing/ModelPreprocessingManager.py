import csv
from io import StringIO

import flask

from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter
from bokeh.colors import red
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.util.string import encode_utf8
from flask import  render_template, redirect, url_for, request, jsonify, app
from flask import send_file
from flask import send_from_directory
from flask.blueprints import Blueprint
import numpy as np
import pandas as pd
import pickle
import numpy as np

from sklearn.preprocessing import Imputer

from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn
from werkzeug.utils import secure_filename

from Config import Config
import os

from ModelPreprocessing.DossierTravailDb import DossierTravail
from ModelPreprocessing.FichierDb import Fichier, db

model = Blueprint('preprocessing_blueprint', __name__, template_folder='templates',
                  url_prefix='/model', static_folder='static')


def sauvegarderData(data,name):
    # save the classifier
    with open(Config.UPLOAD_FOLDER+'data'+name+'.pkl', 'wb') as fid:
        pickle.dump(data, fid)
def loadData(name):
    # load it again
    with open(Config.UPLOAD_FOLDER+'data'+name+'.pkl', 'rb') as fid:
        data = pickle.load(fid)
        return data

@model.route('/upload', methods=['POST','GET'])
#@login_required
def upload():
    #if request.method == 'POST':
    if request.method == 'GET':
        return render_template("upload.html")  #Summarising Groups in the DataFrame
    # Get the name of the uploaded file
    if request.method == 'POST':
     file = request.files['file']
     # Check if the file is one of the allowed types/extensions
     if file :
        # Make the filename safe, remove unsupported chars
        filename = secure_filename(file.filename)
        # Move the file form the temporal folder to
        # the upload folder we setup
        file.save(os.path.join(Config.UPLOAD_FOLDER, filename))
        fichier = Fichier(nomFichier=filename, taille=os.path.getsize(os.path.join(Config.UPLOAD_FOLDER, filename)))
        db.session.add(fichier)
        db.session.commit()
        dossier=DossierTravail(url=Config.UPLOAD_FOLDER)
        db.session.add(dossier)
        db.session.commit()
        try:
         dataFrame = pd.read_csv(os.path.join(Config.UPLOAD_FOLDER, filename), header=0, low_memory=False,
                                    # index_col='Serial',
                                na_values=["97", "94", "98", "95", "U"])
        except:
            dataFrame = pd.read_csv(os.path.join(Config.UPLOAD_FOLDER, filename), sep=';',
                             encoding='latin-1',
                             low_memory=False, error_bad_lines=False, header=0)
        sauvegarderData(dataFrame, filename)
        # Redirect the user to the uploaded_file route, which
        # will basicaly show on the browser the uploaded file
        return redirect(url_for('preprocessing_blueprint.Afficher_data',filename=filename))

'''
dataFrame = pd.read_csv("/Users/USER/PycharmProjects/MarketResearch/data/CocpitW6.csv", header=0, low_memory=False,
                        index_col='Serial',
                        na_values=["97", "94", "98", "95"])

for col_name in dataFrame.columns:

        # dataFrame.replace('0', 'None', inplace=True)
        # dataFrame.replace('U', 'None', inplace=True)
         data=dataFrame.replace("97", '94', '98', np.nan)
         data=dataFrame.replace("95", np.nan)
liste=data.columns.values.tolist()
print(liste)

'''
def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]



@model.route('/AfficherData/<string:filename>',methods=['GET','POST'])
def Afficher_data (filename):
 dataFrame = loadData(filename)

 for col_name in dataFrame.columns:
        # dataFrame.replace('0', 'None', inplace=True)

        data = dataFrame.replace("97", '94', '98', np.nan)
        data = dataFrame.replace("95", np.nan)
        data = dataFrame.replace("U", np.nan)
 liste = data.columns.values.tolist()
 print('ghhh')
 sauvegarderData(data, filename)
 source = ColumnDataSource(data)
 columns = [
    TableColumn(field=c, title=c,width=100) for c in data.columns
     ]
 data_table = DataTable(source=source, columns=columns ,editable=True,width = 4000,fit_columns=True)
 script, div = components(data_table)

 return render_template("affData.html", script=script, div=div, bokeh_css=CDN.render_css(),
                        bokeh_js=CDN.render_js(),data=data,filename=filename)

@model.route('/fillingMissingData/<string:filename>',methods=['GET'])
def filling_data (filename):
     data = loadData(filename)
     global newdf
     numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
     newdf = data.select_dtypes(include=numerics)
     print('les colonnes',newdf.columns.values.tolist())
     # liste=newdf.columns.values.tolist()
     Y = np.array(newdf)
     methode = request.args.get('methode')
    #X = np.array([[23.56], [53.45], ['NaN'], [44.44], [77.78], ['NaN'], [234.44], [11.33], [79.87]]) #recup array of values from
     if methode=='mean' :
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0,copy=False)
     else:
      if methode=='most_frequent':
        imp = Imputer(missing_values='NaN', strategy='most_frequent', axis=0, copy=False)
      else:
        imp = Imputer(missing_values='NaN', strategy='median', axis=0, copy=False)
     Y=imp.fit_transform(Y)
     listt = list(newdf.columns.values)
     listt = [x for x in listt if x != 'Serial']
     print('datacol', listt)
     print('yyyy', Y.shape)
     new_dataFrame = pd.DataFrame(Y, columns=listt)
     source = ColumnDataSource(new_dataFrame)
     columns = [
         TableColumn(field=c, title=c, width=100) for c in new_dataFrame.columns
         ]
     data_table = DataTable(source=source, columns=columns, editable=True, width=4000, fit_columns=True)
     script, div = components(data_table)

     sauvegarderData(new_dataFrame, filename)
     return redirect(url_for('preprocessing_blueprint.Afficher_data', filename=filename))

@model.route('/filling/<string:filename>', methods=['POST'])
def filling (filename):
      print('test')
      methode= request.form['methode']
      print('notre methode c ',methode)
      return redirect(url_for('preprocessing_blueprint.filling_data',methode=methode,filename=filename))
@model.route('/fillup/<string:filename>',methods =['GET'])
def fillup (filename):
     return render_template("filling.html",filename=filename)

@model.route('/Chart/<string:filename>', methods=['get','POST'])
def Chart(filename):
     return render_template("Charts.html",filename=filename)
@model.route('/visualiserSummary/<string:filename>',methods=['GET', 'POST'])
def summary(filename):
 data = loadData(filename)
 mylist = list(data.select_dtypes(include=['int64']).columns)
 data.index.name = None
 datas=data.describe()
 correlation=data.corr().abs()
 indices = np.where(correlation > 0.8)
 indices = [(correlation.columns[x], correlation.columns[y]) for x, y in zip(*indices)
            if x != y and x < y]
 print ('les indices',indices)
 return render_template("sum.html",datas=[datas.to_html(classes=['myList', "table table-hover"])],
                        mylist=mylist,correlation=[correlation.to_html(classes=['myList', "table table-hover"])], corre=correlation,filename=filename)

@model.route('/visualiserCharts/<string:filename>')
def visualiserChart(filename):
    data = loadData(filename)
    args = flask.request.args
    #color = getitem(args, 'color', 'Black')
    categorie = getitem(args,'categorie','Q40B')
    liste = data.columns.values.tolist()
    hist = Histogram(data, values=categorie,
                     title="Age Distribution", bins=10,color=red)
    hist2 = Histogram(data, values="Q40B", label='Q19A_2', color='Q19A_2', legend='top_right',
                      title="Age distribution by prices of Ooredoo", plot_width=400)
    script2,div2=components(hist2)
    script, div = components(hist)

    montant = data['Q35']
    revenue = data['Q45']
    fig = figure(title="nuage de points montant dépensé par rapport aux revenue", x_axis_label='montant',
                 y_axis_label='revenue')
    fig.circle(montant, revenue, size=5, alpha=0.5)
    script3,div3=components(fig)

    return render_template("visualiserChart.html", script3=script3,div3=div3,script=script, div=div,categorie=categorie,data=data,
                           script2=script2,div2=div2, bokeh_css=CDN.render_css(),liste=liste,
                           bokeh_js=CDN.render_js(),filename=filename)



@model.route('/scatterPlot/<string:filename>' ,methods=['GET'])
def scatterPlot(filename):
    data = loadData(filename)
    args = flask.request.args
    #color = getitem(args, 'color', 'Black')
    montant = data[getitem(args,'montant','Q35')]
    revenue=data[getitem(args,'revenue','Q45')]



    fig = figure(title="nuage de points montant dépensé par rapport aux revenue", x_axis_label='montant',
                 y_axis_label='revenue')

    fig.circle(montant, revenue, size=5, alpha=0.5)

    script3,div3=components(fig)
    liste = data.columns.values.tolist()
    return render_template("scatterPlot.html", script3=script3,div3=div3,data=data,montant=montant,filename=filename,
                           revenue=revenue,liste=liste,
                           bokeh_css=CDN.render_css(),
                           bokeh_js=CDN.render_js())

@model.route('/aberrante/<string:filename>')
def aberrante(filename):

        data = loadData(filename)
        # scale data :
        # Select the indices for data points you wish to remove
        from collections import Counter
        outliers_counter = Counter()
        # log_data = np.log(data)


        outliers_scores = None

        # data.drop(['Q53'], axis=1, inplace=True)
        # For each feature find the data points with extreme high or low values
        for feature in data.keys():
            if (data[feature].dtypes != 'object'):
                # TODO: Calculate Q1 (25th percentile of the data) for the given feature

                Q1 = np.percentile(data[feature], 25)

                # TODO: Calculate Q3 (75th percentile of the data) for the given feature
                Q3 = np.percentile(data[feature], 75)

                # TODO: Use the interquartile range to calculate an outlier step (1.5 times the interquartile range)
                step = 1.5 * (Q3 - Q1)

                zeros = np.zeros(len(data[feature]))
                above = data[feature].values - Q3 - step
                below = data[feature].values - Q1 + step
                current_outliers_scores = np.array(np.maximum(zeros, above) - np.minimum(zeros, below)).reshape([-1, 1])
                outliers_scores = current_outliers_scores if outliers_scores is None else np.hstack(
                    [outliers_scores, current_outliers_scores])

                # Display the outliers
                print("Data points considered outliers for the feature '{}':".format(feature))
                current_outliers = data[~((data[feature] >= Q1 - step) & (data[feature] <= Q3 + step))]
                # display(current_outliers)
                outliers_counter.update(current_outliers.index.values)

        # OPTIONAL: Select the indices for data points you wish to remove
        min_outliers_count = 100
        outliers = [x[0] for x in outliers_counter.items() if x[1] >= min_outliers_count]
        print("Data points considered outlier for more than 1 feature: {}".format(outliers))
        ma = [x[1] for x in outliers_counter.items() if x[1] >= min_outliers_count]
        # Remove the outliers, if any were specified
        # outliers=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33]
        good_data = data.drop(data.index[outliers]).reset_index(drop=True)
        print('good', data)
        print('good', good_data)
        sauvegarderData(good_data, filename)
        return redirect(url_for('preprocessing_blueprint.Afficher_data', filename=filename))

    #scale data :

@model.route("/enregistrerDataSet/<string:filename>", methods=['GET', 'POST'])
def enregistrerDataSet(filename):
    data=loadData(filename)
    filename=os.path.join(Config.UPLOAD_FOLDER, filename)
    data.to_csv(filename, sep='\t', encoding='utf-8')
    db.session.add(Fichier(nomFichier=filename,taille=len(data)))
    db.session.commit()
    return render_template("affData.html",filename=filename)