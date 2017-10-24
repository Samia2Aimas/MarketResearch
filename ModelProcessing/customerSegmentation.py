import flask
import pandas as pd
import numpy as np
#from bokeh.core.tests.test_query import plot
from bokeh.charts import Histogram
from bokeh.colors import red
from bokeh.embed import components
from bokeh.models import BoxSelectTool, HoverTool
from flask import render_template
from flask.blueprints import Blueprint
from bokeh.resources import CDN
from sklearn import metrics
from sqlalchemy import column
import pickle
from sklearn.cluster import KMeans

model3 = Blueprint('preprocessing3_blueprint', __name__, template_folder='templates', url_prefix='/model2', static_folder='static')

df = pd.read_csv("C:/Users/USER/PycharmProjects/MarketResearch/data/PR_3G_T1_2017  .csv", sep=';', encoding='latin-1',
                     low_memory=False, error_bad_lines=False, header=0)
print('df head ', df.columns.values.tolist())
print('df', df.shape)
df['n'] = 1
# create a "pivot table" which will give us the number of times each
# customer responded to a given variable
matrix = df.pivot_table(index=[df.index], columns=['A1'], values='n')
# a little tidying up. fill NA values with 0 and make the index into a column
matrix = matrix.fillna(0).reset_index()
x_cols = matrix.columns[1:]

cluster = KMeans(n_clusters=10)
# slice matrix so we only include the 0/1 indicator columns in the clustering
matrix['cluster'] = cluster.fit_predict(matrix[x_cols])
matrix.cluster.value_counts()

@model3.route('/segmentationModel/<int:n_clusters>')
def segmentationModel(n_clusters,columns):
    df = pd.read_csv("C:/Users/USER/PycharmProjects/MarketResearch/data/PR_3G_T1_2017  .csv", sep=';', encoding='latin-1',
                     low_memory=False, error_bad_lines=False, header=0)
    print('df head ', df.columns.values.tolist())
    print('dff', df.shape)
    df['n'] = 1
    # create a "pivot table" which will give us the number of times each
    # customer responded to a given variable
    matrix = df.pivot_table(index=[df.index], columns=columns, values='n')
    # a little tidying up. fill NA values with 0 and make the index into a column
    matrix = matrix.fillna(0).reset_index()
    x_cols = matrix.columns[1:]


    cluster = KMeans(n_clusters)
    # slice matrix so we only include the 0/1 indicator columns in the clustering
    matrix['cluster'] = cluster.fit_predict(matrix[x_cols])
    matrix.cluster.value_counts()
    #print('homo', metrics.homogeneity_score(matrix, cluster.labels_))

@model3.route('/sauvegarderModele')
def sauvegarderModele(classifier, name):
        # save the classifier
        with open('F:/3cs/My_PFE/version/tmp/modeles' + name + '.pkl', 'wb') as fid:
            pickle.dump(classifier, fid)

def loadModel(name):
        # load it again
        with open('F:/3cs/My_PFE/version/tmp/modeles' + name + '.pkl', 'rb') as fid:
            gnb_loaded = pickle.load(fid)
            return gnb_loaded




@model3.route('/segmentationChart/')
def segmentationChart():
    segmentationModel(10,['A1','F6'])
    df = matrix['cluster']
    TOOLS = [BoxSelectTool(), HoverTool()]
    hist = Histogram(df,values='cluster',
                     title="segmentations des clients ",bins=20,color=red,tools=TOOLS)
    script, div = components(hist)
    return render_template("segmentation.html", script=script, div=div,data=matrix,bokeh_css=CDN.render_css(),bokeh_js=CDN.render_js())
'''
@model3.route(('/homogeneity'))
def homogeneity():
    print('homo',metrics.homogeneity_score(matrix.target, estimator.labels_))
'''

f1 = df['A1'].values
f2 = df.index.values

X= np.matrix(zip(f1, f2))
#kmeans = KMeans(n_clusters=2).fit(X)
to_drop = [ 'PrjName', 'Q31C13O', 'F6O','Q30O','Q37_MO','Q10O','Q42_MO','Q42_DO' ]
numerics = ['int16', 'int32', 'int64', 'float16']
#df = df.select_dtypes(include=numerics)
#kmeans = KMeans(n_clusters=10, random_state=0).fit(df)
#print('kmeans',kmeans)
#print('count', df.cluster.value_counts())
'''
from ggplot import *
ggplot(matrix, aes(x='factor(cluster)')) + geom_bar() + xlab("Cluster") + ylab("Customers\n(# in cluster)")'''

from sklearn.decomposition import PCA

pca = PCA(n_components=2)
matrix['x'] = pca.fit_transform(matrix[x_cols])[:, 0]
matrix['y'] = pca.fit_transform(matrix[x_cols])[:, 1]
matrix = matrix.reset_index()

customer_clusters = matrix[[ 'cluster', 'x', 'y']]
customer_clusters.head()



'''@model3.route('/seg')
def seg():
    plot.circle_cross(x=matrix['x'], y=matrix['y'], size=40, fill_alpha=0, line_width=2, color=['red', 'blue', 'purple'])

    plot.text(text=['setosa', 'versicolor', 'virginica'], x=matrix['x'], y=matrix['y'], text_font_size='30pt')

    i = 0  # counter

    # begin plotting each petal length / width

    # We get our x / y values from the original plot data.

    # The k-means algorithm tells us which 'color' each plot point is,

    # and therefore which group it is a member of.
    '''
'''for sample in petal_data:

        # "labels_" tells us which cluster each plot point is a member of

        if cluster.labels_[i] == 0:

        plot.circle(x=sample[0], y=sample[1], size=15, color="red")

        if cluster.labels_[i] == 1:

        plot.circle(x=sample[0], y=sample[1], size=15, color="blue")

        if cluster.labels_[i] == 2:

        plot.circle(x=sample[0], y=sample[1], size=15, color="purple")

        i += 1

    bokeh.plotting.show(plot)
    return render_template('Confusion_matrix.html', p=p)'''

'''df = pd.merge(df_transactions, customer_clusters)
df = pd.merge(df_offers, df)'''
'''
from ggplot import *

ggplot(df, aes(x='x', y='y', color='cluster')) + \
    geom_point(size=75) + \
    ggtitle("Customers Grouped by Cluster")'''

cluster_centers = pca.transform(cluster.cluster_centers_)
cluster_centers = pd.DataFrame(cluster_centers, columns=['x', 'y'])
cluster_centers['cluster'] = range(0, len(cluster_centers))
'''
ggplot(df, aes(x='x', y='y', color='cluster')) + \
    geom_point(size=75) + \
    geom_point(cluster_centers, size=500) +\
    ggtitle("Customers Grouped by Cluster")
'''

'''df['is_4'] = df.cluster == 4
df.groupby("is_4").varietal.value_counts()
df.groupby("is_4")[['min_qty', 'discount']].mean()'''

