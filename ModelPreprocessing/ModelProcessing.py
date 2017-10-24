from pyexpat import features


import numpy as np
#from sklearn.cross_validation import KFold, train_test_split
import subprocess
import mpld3
import statsmodels.api as sm
import statsmodels.formula.api as smf
from bokeh.charts import HeatMap, show
from bokeh.embed import components
from bokeh.models import ColumnDataSource, TableColumn
from sklearn.feature_selection import RFE
from bokeh.resources import CDN
from flask import redirect
from flask import render_template
from flask import request
from flask import session
from flask import url_for
import seaborn as sns
from pip.req.req_file import preprocess
from sklearn.cross_validation import cross_val_score
from sklearn.externals import joblib
from sklearn.feature_selection import RFE
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from bokeh.plotting import *

from sklearn.model_selection import KFold
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier as RF
from sklearn.neighbors import KNeighborsClassifier as KNN

from sklearn.metrics import confusion_matrix
from sklearn import cross_validation, metrics, svm, tree, model_selection
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import DecisionTreeRegressor

from bokeh.models.widgets import DataTable, DateFormatter, TableColumn

from sklearn.preprocessing import StandardScaler
from flask.blueprints import Blueprint
import pandas as pd
from sklearn.tree import export_graphviz
#import matplotlib as plt
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from statsmodels.regression.linear_model import OLSResults

from ModelProcessing.performanceDB import Performance, db

model2 = Blueprint('preprocessing2_blueprint', __name__, template_folder='templates', url_prefix='/model2', static_folder='static')

dataFrame = pd.read_csv("/Users/USER/PycharmProjects//MarketResearch/data/EXP_GAT_26_Mars_2017_.csv", header=0, low_memory=False,error_bad_lines=False ,na_values = ["97", "94", "99", "95"]
)

'''

        # dataFrame.replace('0', 'None', inplace=True)
        # dataFrame.replace('U', 'None', inplace=True)
         data=dataFrame.replace("97", '94', '99', np.nan)
         data=dataFrame.replace("95", np.nan)'''




churn_result=dataFrame['Q11']
churn_class=dataFrame['QGC']


#liste=dataFrame_space.columns.values.tolist()

#print(liste)
#'prepare data and split

import pickle
def sauvegarderModele(classifier,name):
    # save the classifier
    with open('F:/3cs/My_PFE/version/tmp/modeles'+name+'.pkl', 'wb') as fid:
        pickle.dump(classifier, fid)
def loadModel(name):
    # load it again
    with open('F:/3cs/My_PFE/version/tmp/modeles'+name+'.pkl', 'rb') as fid:
        gnb_loaded = pickle.load(fid)
        return gnb_loaded


@model2.route('/selectVariables', methods=['POST'])
def selectVariables ():
      multiselect = request.form.getlist('mymultiselect')
      print('my multiselect',multiselect)
      dataFrame_space = dataFrame.drop(multiselect, axis=1)
      churn_result = dataFrame['Q11']
      yy=np.where(churn_result==1 ,1,0)
      numerics = ['int16', 'int32', 'int64', 'float16']
      dataFrame_space = dataFrame_space.select_dtypes(include=numerics)
      XX = dataFrame_space.as_matrix().astype(np.float)
      scaler = StandardScaler()
      XX = scaler.fit_transform(XX)
      clf = svm.SVC(probability=True, verbose=True)


      SVM = accuracy(yy, run_cv(XX, yy, SVC))
      RFF = accuracy(yy, run_cv(XX, yy, RF))
      KN = accuracy(yy, run_cv(XX, yy, KNN))
      from sklearn.metrics import classification_report
      metrics_SVM = classification_report(yy, run_cv(XX, yy, SVC))
      metrics_RFF = classification_report(yy, run_cv(XX, yy, RF))
      metrics_KN = classification_report(yy, run_cv(XX, yy, KNN))
      matrice=confusion_matrix(yy, run_cv(XX, yy, SVC))
      tn = matrice[0][0]
      # print('tn',tn)
      fp = matrice[0][1]
      fn = matrice[1][0]
      tp = matrice[1][1]
      #performance = Performance(tp=tp,tn=tn,fp=fp,fn=fn)
      #db.session.add(performance)
      #db.session.commit()


      return redirect(url_for('preprocessing2_blueprint.construireModel',SVM=SVM,RFF=RFF,KN=KN,
                                                                          metrics_KN=metrics_KN,metrics_SVM=metrics_SVM,
                                                                          metrics_RFF=metrics_RFF,matrice=matrice
                              ,tp=tp,tn=tn,fp=fp,fn=fn))

@model2.route('/construireModel',methods=['GET','POST'])
def construireModel ():

        SVM = request.args.get('SVM')
        RFF = request.args.get('RFF')
        KN = request.args.get('KN')
        matrice=request.args.get('matrice')
        tn = request.args.get('tn')
        # print('tn',tn)
        fp = request.args.get('fp')
        fn = request.args.get('fn')
        tp = request.args.get('tp')
        metrics_KN = request.args.get('metrics_KN')
        metrics_SVM = request.args.get('metrics_SVM')
        metrics_RFF = request.args.get('metrics_RFF')
        session['metrics_KN'] = metrics_KN
        session['metrics_SVM'] = metrics_SVM
        session['metrics_RFF'] = metrics_RFF
        session['matrice'] =matrice
        session['tp'] = tp
        session['tn'] = tn
        session['fp'] = fp
        session['fn'] = fn
        return render_template("construireModel.html", SVM=SVM , RFF=RFF ,KN=KN,metrics_KN=metrics_KN,matrice=matrice,
                               metrics_SVM=metrics_SVM,metrics_RFF=metrics_RFF)



@model2.route('/classificationReport', methods=['GET'])
def classificationReport():

    report_data = []
    report_data1 = []
    report_data2= []

    metrics_KN= session['metrics_KN']

    metrics_SVM= session['metrics_SVM']

    metrics_RFF= session['metrics_RFF']

    lines = metrics_KN.split('\n')
    lines1= metrics_SVM.split('\n')
    lines2= metrics_RFF.split('\n')
    for line in lines[2:-3]:
        row = {}
        row_data = line.split('      ')
        row['class'] = row_data[0]
        row['precision'] = float(row_data[1])
        row['recall'] = float(row_data[2])
        row['f1_score'] = float(row_data[3])
        row['support'] = float(row_data[4])

        report_data.append(row)
    for line in lines1[2:-3]:
            row = {}
            row_data = line.split('      ')
            row['class'] = row_data[0]
            row['precision'] = float(row_data[1])
            row['recall'] = float(row_data[2])
            row['f1_score'] = float(row_data[3])
            row['support'] = float(row_data[4])
            report_data1.append(row)
    for line in lines2[2:-3]:
                row = {}
                row_data = line.split('      ')
                row['class'] = row_data[0]
                row['precision'] = float(row_data[1])
                row['recall'] = float(row_data[2])
                row['f1_score'] = float(row_data[3])
                row['support'] = float(row_data[4])
                report_data2.append(row)
    dataframe = pd.DataFrame.from_dict(report_data)
    dataframe1 = pd.DataFrame.from_dict(report_data1)
    dataframe2 = pd.DataFrame.from_dict(report_data2)
    source = ColumnDataSource(dataframe)
    columns = [
        TableColumn(field=c, title=c, width=100) for c in dataframe.columns
        ]
    data_table = DataTable(source=source, columns=columns, fit_columns=True)
    script, div = components(data_table)

    source = ColumnDataSource(dataframe)
    columns = [
        TableColumn(field=c, title=c, width=100) for c in dataframe1.columns
        ]
    data_table1 = DataTable(source=source, columns=columns, fit_columns=True)
    script1, div1 = components(data_table1)

    source = ColumnDataSource(dataframe)
    columns = [
        TableColumn(field=c, title=c, width=100) for c in dataframe2.columns
        ]
    data_table2 = DataTable(source=source, columns=columns, fit_columns=True)
    script2, div2 = components(data_table2)

    return render_template("classificationReport.html", script=script, div=div, bokeh_css=CDN.render_css(),
                           script2=script2, div2=div2,script1=script1, div1=div1,
                           bokeh_js=CDN.render_js(), data_table=data_table,data_table1=data_table1,data_table2=data_table2)


@model2.route('/select')
def select ():
    liste=dataFrame.columns.values.tolist()
    return render_template("SelectVariables.html",liste=liste)

def run_cv(XX,yy,clf_class,**kwargs):
    # Construct a kfolds object
    from sklearn.cross_validation import   KFold
    kf = KFold(len(yy),n_folds=5,shuffle=True)
    yy_pred = yy.copy()
    y_prob=yy.copy()

    # Iterate through folds
    for train_index_, test_index_, in kf:
        #XX_train, XX_test = XX[train_index_], XX[test_index_]
        XX_train, XX_test, yy_train, yy_test = train_test_split(XX, yy, test_size=0.25, random_state=42)
        #yy_train = yy[train_index_]
        # Initialize a classifier with key word arguments
        clf = clf_class(**kwargs)

        clf.fit(XX_train,yy_train)

        clf.probability=True

        #y_prob[test_index_] = clf.predict_proba(XX_test)
        yy_pred[test_index_] = clf.predict(XX_test)

    return yy_pred

def accuracy(y_true,y_pred):
    # NumPy interpretes True and False as 1. and 0.
    return np.mean(y_true == y_pred)



data = pd.read_csv("/Users/USER/PycharmProjects//MarketResearch/data/Book.csv", header=0, low_memory=False,error_bad_lines=False ,na_values = ["97", "94", "99", "95"]
)
#print(data.columns.values.tolist())



@model2.route('/selectSatisfaction')
def selectSatisfaction():
    liste = data.columns.values.tolist()
    return render_template("SelectSatisfaction.html", liste=liste)

@model2.route('/selectVariablesSatisfaction', methods=['POST'])
def selectVariablesSatisfaction ():
    colonneNps=request.form['colonneNps']

    multiselectDrop = request.form.getlist('mymultiselectDrop')
    multiselectBin = request.form.getlist('mymultiselectBin')
    multiselectSansBin= request.form.getlist('mymultiselectSansBin')

    data_space = data.drop(multiselectDrop, axis=1)
    np.seterr(invalid='ignore')
    nps = data[colonneNps]
    # data_space.set_value(i, 'typeNps', 0)
    typeNps = []
    for row in nps:
        if row <= 6:
            typeNps.append(0)
        else:
            if row >= 9:
                typeNps.append(2)
            else:
                typeNps.append(1)

    data_space['typeNps'] = typeNps

    #col_to_dummy = ['Q7_PRINSIM', 'Q8_OFFRESIM', 'Q12_TESTDATA', 'Q12_TESTCALLCENTER', 'Q12_TESTSHOP', 'Q12_TESTRECLAM',
     #               'Q12_TESTSITEWEB', 'Q25_SEXE', 'Q26_JOBACTUEL', 'Q27_NIVETUDES', 'typeNps']
    col_to_dummy= multiselectBin
    print("these are the col to dummy",col_to_dummy)
    #col_to_keep = ['Q9_ARPU', 'Q29_COMMUNE', 'Q28_WILAYA']
    col_to_keep=multiselectSansBin
    print("these are the col to keeep",col_to_keep)
    # frames=[]
    result = data_space[col_to_keep]

    data_dummy = pd.get_dummies(data_space['typeNps'], prefix='typeNps')
    result = result.join(data_dummy)
    for col in data_space[col_to_dummy]:
        data_dummy = pd.get_dummies(data_space[col], prefix=col)
        result = result.join(data_dummy)
        # frames.append(data_dummy)
    detract = result['typeNps_2']
    to_drop = ['typeNps_0', 'typeNps_1', 'typeNps_2']
    resultat = result.drop(to_drop, axis=1)
    resultat['Intercept'] = 1.0
    # dta = data_space[col_to_keep].join(frames.ix[:, 'Q7_PRINSIM_1':])
    print("le dataframe looks like ", resultat)

    #X_train, X_test, y_train, y_test = model_selection.train_test_split(resultat,detract,
     #                                                                      test_size=0.25, random_state=7)
    #X_train=pd.DataFrame(X_train),
    #y_train=pd.DataFrame(y_train)
    logit = sm.Logit(detract,resultat,missing='drop')
    # fit the model
    result = logit.fit_regularized()
    #y_pred = result.predict(X_test
     #                       )
    #print(" this is my predictiton" ,y_pred)
    filename='/Users/USER/PycharmProjects/MarketResearch/SavedModels/logistic_model.pickle'
    #pickle._dump(result, open(filename,'wb'))
    result.save(filename)
    new_result=OLSResults.load(filename)

    summary=result.summary()
    print("Summary :")
    print(summary)
    print("les intervals de confiances :")
    int_conf=result.conf_int()
    print(int_conf)
    params=result.params
    print("mes params",params)
    # odds ratios only
    print("odds ratios only :")
    odds=np.exp(result.params)
    print(odds)
    return redirect(url_for('preprocessing2_blueprint.construireModelSatisfaction', params=params,summary=summary,
                            int_conf=int_conf,odds=odds,new_result=new_result))


@model2.route('/construireModelSatisfaction',methods=['GET','POST'])
def construireModelSatisfaction ():

        summary = request.args.get('summary')
        int_conf = request.args.get('int_conf')
        params=request.args.get('params')
        odds = request.args.get('odds')


        return render_template("construireModelSatisfaction.html",odds=odds,params=params,summary=summary
                               ,int_conf=int_conf)

@model2.route('/nouvellePredicitionSatisfaction',methods=['GET'])
def nouvelleSatisfaction ():

    loaded_model=sm.load("/Users/USER/PycharmProjects/MarketResearch/SavedModels/logistic_model.pickle")
    result=loaded_model.score(Y_test,X_test)
    #print ("my result is ")
    #print(result)


############le model Revenue territory #########################################


dataFrame2 = pd.read_csv("/Users/USER/PycharmProjects/MarketResearch/data/PR_3G_T1_2017  .csv", header=0, low_memory=False,
                        sep=";",encoding='latin-1',
                        na_values=["97", "94", "98", "95"])

####select la colonne , des wilayas ######


@model2.route('/selectRegression')
def selectRegression():
    liste = dataFrame2.columns.values.tolist()
    return render_template("SelectRegression.html", liste=liste)



@model2.route('/selectVariablesRegression', methods=['POST'])
def selectVariablesRegression ():
      colonneVille= request.form['colonneVille']
      colonneRevenue=request.form['colonneRevenue']
      multiselect = request.form.getlist('mymultiselect')
      print ('mymultiselect is ',multiselect)
      dataFrame_wilaya = dataFrame2.groupby(colonneVille, as_index=False)
      sumi = dataFrame_wilaya.aggregate({ var:np.sum for var in multiselect})
      sumi_input=sumi
      print(sumi)
      to_drop=[colonneRevenue,colonneVille]
      sumi_output = sumi.drop(to_drop, axis=1)
      X_train, X_test, y_train, y_test = model_selection.train_test_split(sumi_output,sumi_input[colonneRevenue],
                                                                          test_size=0.25, random_state=7)
      #lm = smf.ols( formula= colonneRevenue ~ multiselect, data=sumi).fit()

      #lm = sm.OLS(sumi_input[colonneRevenue],sumi_output).fit()

      lm = sm.OLS(y_train, X_train).fit()

      filename = '/Users/USER/PycharmProjects/MarketResearch/SavedModels/Regression_model.pickle'
      # pickle._dump(result, open(filename,'wb'))
      lm.save(filename)
      y_pred = lm.predict(X_test
                             )

      print("my summary is : ")
      summary=lm.summary()
      print(summary)

      print("my parameteres : ")
      params=lm.params
      print(params)
      print("intervalle de confaince à 95%: ")
      int_confiance=lm.conf_int()
      print(int_confiance)
      print("parameters significance  :")
      param_significance=lm.pvalues
      print(lm.pvalues)
###  evaluation of the model (mean squad error , rersidual error####
      print("squad  error : ")
      squad=lm.rsquared
      print(lm.rsquared)

      return redirect(url_for('preprocessing2_blueprint.construireModelRegression',summary=summary.tables[0].as_html(),
                              params=params, int_confiance=int_confiance,y_pred=y_pred
                              ,sumi_output=sumi_output,squad=squad, lm=lm , param_significance=param_significance,))

@model2.route('/construireModelRegression',methods=['GET','POST'])
def construireModelRegression ():
        summary = request.args.get('summary')
        params = request.args.get('params')
        int_confiance = request.args.get('int_confiance')
        param_significance= request.args.get('param_significance')
        squad = request.args.get('squad')
        lm=request.args.get('lm')
        y_pred = request.args.get('y_pred')
        session['y_pred']=y_pred
        return render_template("construireModelRegression.html", summary=summary, params=params, int_confiance=int_confiance,
                             squad=squad, param_significance=param_significance)


@model2.route('/nouvelleRegression',methods=['GET'])
def nouvelleRegression ():
    y_pred=session['y_pred']

    dataframe=pd.DataFrame.from_records(y_pred,columns=['revenue'])

    source = ColumnDataSource(dataframe)
    columns = [
        TableColumn(field="revenue", title="revenue", width=100)
        ]
    data_table = DataTable(source=source, columns=columns, width=4000, fit_columns=True)
    script, div = components(data_table)
    return render_template("nouvelleRegression.html",script=script,div=div,bokeh_css=CDN.render_css(),
                           bokeh_js=CDN.render_js(), data_table=data_table)


@model2.route('/interpretation',methods=['GET','POST'])
def interpretation ():
  summary = session['summary']
  return render_template("interpretation.html" ,summary=summary.tables[0].as_html())
@model2.route('/models')
def models():

    return render_template("models.html")



@model2.route('/pickList')
def pickList():
    liste = dataFrame.columns.values.tolist()
    return render_template("pickList.html" ,liste=liste)
@model2.route('/courbeRegression',methods=['GET'])
def courbeRegression():
 lm=session['lm']
 sumi_output=session['sumi_output']
 '''
 print ("my sumi output" ,sumi_output)
 plt.scatter(sumi_output[0], sumi_output.colonneRevenue, alpha=0.3)
 plt.xlabel('Revenue Voice')
 plt.ylabel('Revenue Total')
 income_linspace = np.linspace(sumi_output.Q12.min(), sumi_output.Q12, 100)

 plt.plot(income_linspace, lm.params[0] + lm.params[1] * income_linspace + lm.params[2] * 0, 'r')
 plt.plot(income_linspace, lm.params[0] + lm.params[1] * income_linspace + lm.params[2] * 1, 'g')
 plt.show()
'''
 fig, ax = plt.subplots()
 fig = sm.graphics.plot_fit(lm, sumi_output.params[0], ax=ax)
 ax.set_ylabel("Revenue ")
 ax.set_xlabel("revenue 3G")
 ax.set_title("Linear Regression")
 plt.show()
# Specify that the two graphs should be on the same plot.

#scatter(x, y, marker="square", color="blue")
#show()


@model2.route('/show_confusion_matrix', methods=['GET','POST'])
def show_confusion_matrix():
    print('Ccc')
    #C=session['matrice']
    print('la matrice')
    #print(C)

    class_labels=['0', '1']
    """
    C: ndarray, shape (2,2) as given by scikit-learn confusion_matrix function
    class_labels: list of strings, default simply labels 0 and 1.

    Draws confusion matrix with associated metrics.
    """
    import matplotlib.pyplot as plt

    import numpy as np
    print('hee')
    #assert C.shape == (2, 2), "Confusion matrix should be from binary classification only."

    # true negative, false positive, etc...
    '''tn = C[0][ 0]
    #print('tn',tn)
    fp = C[0][ 1]
    fn = C[1][0]
    tp = C[1][ 1]'''

    tn = int(session['tn'])
    # print('tn',tn)
    fp =int(session['fp'])
    fn = int(session['fn'])
    tp = int(session['tp'])
    C=np.array([[int(tn),int(fp)],[int(fn),int(tp)]])
    print ('ccc',C)
    '''C[0][0]=tn
    # print('tn',tn)
    C[0][1]=fp
    C[1][0]=fn
    C[1][1]=tp'''
    NP = fn + tp  # Num positive examples
    NN = tn + fp  # Num negative examples
    N = NP + NN

    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111)
    ax.imshow(C, interpolation='nearest', cmap=plt.cm.get_cmap('RdBu'))

    # Draw the grid boxes
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(2.5, -0.5)
    ax.plot([-0.5, 2.5], [0.5, 0.5], '-k', lw=2)
    ax.plot([-0.5, 2.5], [1.5, 1.5], '-k', lw=2)
    ax.plot([0.5, 0.5], [-0.5, 2.5], '-k', lw=2)
    ax.plot([1.5, 1.5], [-0.5, 2.5], '-k', lw=2)

    # Set xlabels
    ax.set_xlabel('Predicted Label', fontsize=16)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(class_labels + [''])
    ax.xaxis.set_label_position('top')
    ax.xaxis.tick_top()
    # These coordinate might require some tinkering. Ditto for y, below.
    ax.xaxis.set_label_coords(0.34, 1.06)

    # Set ylabels
    ax.set_ylabel('True Label', fontsize=16, rotation=90)
    ax.set_yticklabels(class_labels + [''], rotation=90)
    ax.set_yticks([0, 1, 2])
    ax.yaxis.set_label_coords(-0.09, 0.65)

    # Fill in initial metrics: tp, tn, etc...
    ax.text(0, 0,
            'True Neg: %d\n(Num Neg: %d)' % (tn, NN),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 1,
            'False Neg: %d' % fn,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 0,
            'False Pos: %d' % fp,
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(1, 1,
            'True Pos: %d\n(Num Pos: %d)' % (tp, NP),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    # Fill in secondary metrics: accuracy, true pos rate, etc...
    ax.text(2, 0,
            'False Pos Rate: %.2f' % (fp / (fp + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 1,
            'True Pos Rate: %.2f' % (tp / (tp + fn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(2, 2,
            'Accuracy: %.2f' % ((tp + tn + 0.) / N),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    ax.text(0, 2,
            'Neg Pre Val: %.2f' % (1 - fn / (fn + tn + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))

    '''ax.text(1, 2,
            'Pos Pred Val: %.2f' % (tp / (tp + fp + 0.)),
            va='center',
            ha='center',
            bbox=dict(fc='w', boxstyle='round,pad=1'))'''

    plt.tight_layout()
    html_text = mpld3.fig_to_html(fig)
    #print('html',html_text)
    #plt.show()
    return render_template('Confusion_matrix.html', html_text=html_text)
    return html_text

@model2.route('/matriceConfusion', methods=['GET','POST'])
def matriceConfusion():

    html_text = show_confusion_matrix()#confusion_matrix(yy, run_cv(XX, yy, SVC)), ['Class 0', 'Class 1'])
    #print('httttt',html_text)
    return render_template('Confusion_matrix.html', html_text=html_text)