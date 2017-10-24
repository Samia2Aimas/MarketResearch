
import flask
from bokeh.charts import Bar
from bokeh.charts import BoxPlot
from bokeh.charts import Donut, HeatMap, Histogram, Line, Scatter
from bokeh.colors import red
from bokeh.embed import components
from bokeh.plotting import figure
from bokeh.resources import CDN
from bokeh.util.string import encode_utf8
from flask import  render_template, redirect, url_for, request, jsonify
from flask import send_file
from flask.blueprints import Blueprint
import numpy as np
import pandas as pd
from numpy import genfromtxt
from matplotlib import pyplot as plt

import numpy as np
from pandas.formats.style import Styler
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import Imputer
from bokeh.plotting import output_file
from bokeh.models import ColumnDataSource
from bokeh.models.widgets import DataTable, DateFormatter, TableColumn



view = Blueprint('view_blueprint', __name__, template_folder='templates', url_prefix='/view', static_folder='static')

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


def getitem(obj, item, default):
    if item not in obj:
        return default
    else:
        return obj[item]

@view.route('/visualiserCharts/')
def visualiserChart():

    args = flask.request.args
    #color = getitem(args, 'color', 'Black')
    categorie = getitem(args,'categorie','Q40B')

    hist = Histogram(data, values=categorie,
                     title="Age Distribution", bins=10,color=red)
    hist2 = Histogram(data, values="Q40B", label='Q19A_2', color='Q19A_2', legend='top_right',
                      title="Age distribution by prices of Ooredoo", plot_width=400)
    script, div = components(hist)
    script2,div2=components(hist2)

    montant = data['Q35']
    revenue = data['Q45']
    fig = figure(title="nuage de points montant dépensé par rapport aux revenue", x_axis_label='montant',
                 y_axis_label='revenue')

    fig.circle(montant, revenue, size=5, alpha=0.5)

    script3,div3=components(fig)

    return render_template("visualiserChart.html", script3=script3,div3=div3,script=script, div=div,categorie=categorie,data=data,
                           script2=script2,div2=div2, bokeh_css=CDN.render_css(),
                           bokeh_js=CDN.render_js())



@view.route('/scatterPlot/')
def scatterPlot():
    args = flask.request.args
    #color = getitem(args, 'color', 'Black')
    montant = data[getitem(args,'montant','Q35')]
    revenue=data[getitem(args,'revenue','Q45')]



    fig = figure(title="nuage de points montant dépensé par rapport aux revenue", x_axis_label='montant',
                 y_axis_label='revenue')



    fig.circle(montant, revenue, size=5, alpha=0.5)

    script3,div3=components(fig)

    return render_template("scatterPlot.html", script3=script3,div3=div3,data=data,montant=montant,
                           revenue=revenue,
                           bokeh_css=CDN.render_css(),
                           bokeh_js=CDN.render_js())


