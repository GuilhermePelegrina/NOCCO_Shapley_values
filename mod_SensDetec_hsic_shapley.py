''' Modules used NOCCO Shapley values analysis, for detection of relevant and disparity prone features'''

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import plotly.io as pio
pio.renderers.default='svg'
import plotly.graph_objects as go
from sklearn.preprocessing import OneHotEncoder
from math import comb
from itertools import chain, combinations
import sklearn.datasets 
import random


from sklearn.metrics.pairwise import rbf_kernel


def func_read_data(data_imp):
    ''' This function reads the considered dataset '''
    
    if data_imp == 'adult':
        # Adult dataset
        # We removed education as it reflects educational-num
        # We removed fnlwgt according to paper's suggestion
        # We considered age={25–60, <25 or >60}
        # We considered workclass={private,non-private}
        # We considered marital-status={married,never-married,other}
        # We considered occupation={office,heavy-work,service,other}
        # We considered race={white,non-white}
        # We considered native-country={US,non-US}

        dataset = pd.read_csv('data_adult.csv', na_values='?').dropna()
        X = dataset.iloc[:, 0:-1]
        X = X.drop('fnlwgt',axis=1)
        X = X.drop('education',axis=1)
        age1, age2, age3 = X.age<25, X.age.between(25, 60), X.age>60
        X['age'] = np.select([age1, age2, age3], ['<25', '25-60', '>60'], default=None)
        X['workclass'] = np.where(X['workclass'] != 'Private', 'Non-private', X['workclass'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-civ-spouse', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-spouse-absent', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Married-AF-spouse', 'married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Never-married', 'never-married', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Divorced', 'other', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Separated', 'other', X['marital-status'])
        X['marital-status'] = np.where(X['marital-status'] == 'Widowed', 'other', X['marital-status'])
        X['occupation'] = np.where(X['occupation'] == 'Adm-clerical', 'office', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Exec-managerial', 'office', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Craft-repair', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Farming-fishing', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Handlers-cleaners', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Machine-op-inspct', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Transport-moving', 'heavy-work', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Other-service', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Priv-house-serv', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Protective-serv', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Tech-support', 'service', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Prof-specialty', 'other', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Armed-Forces', 'other', X['occupation'])
        X['occupation'] = np.where(X['occupation'] == 'Sales', 'other', X['occupation'])
        #X['race'] = np.where(X['race'] != 'White', 'Non-white', X['race'])
        X['native-country'] = np.where(X['native-country'] != 'United-States', 'Non-United-States', X['native-country'])
        
        #sensitive = ['age', 'marital_status', 'relationship', 'race', 'gender', 'native-country']
        #X = X.drop('native-country',axis=1)
        #X = X.drop('workclass',axis=1)
        
        y =  dataset.loc[:,'income']=='>50K'
        y = 2*y.astype(int)-1
            
    if data_imp == 'compas':
        # Compas dataset
        
        dataset = pd.read_excel('data_compas.xlsx')
        X = dataset.iloc[:, 0:8]
        X['race'] = np.where(X['race'] == 'Hispanic', 'Other', X['race'])
        X['race'] = np.where(X['race'] == 'Asian', 'Other', X['race'])
        X['race'] = np.where(X['race'] == 'Native American', 'Other', X['race'])
        
        #sensitive = ['sex', 'age_cat', 'race']
        
        y =  dataset.loc[:,'score_risk']
        y = 2*y-1
        
    if data_imp == 'lsac_new':
        # LSAC dataset (new)
          
        dataset = pd.read_csv('data_lsac_new.csv')
        dataset[['fulltime','male','race']]=dataset[['fulltime','male','race']].astype(str)
        X = dataset.iloc[:, 0:-1]
        
        #sensitive = ['male', 'race']
        
        y =  dataset.loc[:,'pass_bar']
        y = 2*y-1
        
    if data_imp == 'rice':
        # Rice (Commeo - 1 and Osmancik - 0) dataset
        dataset = pd.read_excel('data_rice.xlsx')
        X = dataset.loc[:, dataset.columns!='Class']
        vals = dataset.values
        y = 2*vals[:,-1].astype(int)-1
        
    if data_imp == 'banknotes':
        # Bank notes dataset
        dataset = pd.read_csv('banknotes.csv')
        vals = dataset.values
        X = dataset.loc[:, dataset.columns!='authentic']
        y = 2*vals[:,4].astype(int)-1
        
    if data_imp == 'redwine':
        # Red wine quality dataset
        dataset = pd.read_csv('data_wine_quality_red.csv')
        X = dataset.drop('quality', axis=1)
        y = dataset['quality']
        y = 2*(y>5).astype(int)-1
        
    if data_imp == 'diabetesPima':
        # Diabetes (PIMA) dataset "
        dataset = pd.read_csv('data_diabetes_pima.csv')
        vals = dataset.values
        X = dataset.drop('Outcome', axis=1)
        y = 2*dataset['Outcome'].astype(int)-1
        
    if data_imp == 'raisin':
        # Raisin dataset "
        dataset = pd.read_excel('data_raisin.xlsx')
        X = dataset.drop('Class', axis=1)
        y = 2*(dataset['Class']=='Kecimen').astype(int)-1
    
    if data_imp == 'random':
        w = np.array([0.25, 0.40, 0, 0.15, 0.2])
        X = np.random.uniform(0,1,[3000,5])
        X[:,4] = np.copy(X[:,3])
        y = w[0]*X[:,0]+w[1]*X[:,1]+w[2]*X[:,2]+w[3]*X[:,3]+w[4]*X[:,4]+0.01*np.random.rand(3000,)
        y = 2*(y>np.mean(y)).astype(int)-1
        
        features_names = list()
        for ii in range(X.shape[1]):
            features_names.append(f'Feature {ii+1}')
        
        X = pd.DataFrame(X, columns = features_names)
        y = pd.Series(y)
        
    # Reducing the number of samples for HSIC / NOCCO calculation (if needed)
    n_samp = X.shape[0]
    if  n_samp > 5000:
        samp = random.sample(range(X.shape[0]), 5000)
        X = X.iloc[samp,:]
        y = y.iloc[samp]
        
    return X,y

def func_sensitive_indices(data_imp):
    if data_imp == 'adult':
        # Adult dataset
        sensitive = [0,3,5,6,7,11]
            
    if data_imp == 'compas':
        # Compas dataset
        sensitive = [0,1,2]
        
    if data_imp == 'lsac_new':
        # LSAC dataset (new)
        sensitive = [8,9]
        
    return sensitive
    
def func_confusion_matrix(y_class,y_pred):
    ' This function calculates the true/false positive/negative (confusion matrix) '
    
    tn = sum((y_class < 0).astype(float) * np.array(y_pred < 0))
    tp = sum((y_class > 0).astype(float) * np.array(y_pred > 0))
    fn = sum((y_class < 0).astype(float) * np.array(y_pred > 0))
    fp = sum((y_class > 0).astype(float) * np.array(y_pred < 0))

    return tn,tp,fn,fp

def func_confusion_matrix_groups(y_class,y_pred,X_aux,features_categorical,features_to_encode_aux,cont):
    ' This function calculates the true/false positive/negative (confusion matrix) given a sensitive group '
    
    tn = sum((y_class < 0).astype(float) * np.array(y_pred < 0) * np.squeeze(np.array(X_aux == features_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
    tp = sum((y_class > 0).astype(float) * np.array(y_pred > 0) * np.squeeze(np.array(X_aux == features_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
    fn = sum((y_class < 0).astype(float) * np.array(y_pred > 0) * np.squeeze(np.array(X_aux == features_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
    fp = sum((y_class > 0).astype(float) * np.array(y_pred < 0) * np.squeeze(np.array(X_aux == features_categorical[cont][len(max(features_to_encode_aux, key=len))+1:])))
    
    return tn,tp,fn,fp

def func_confusion_matrix_coalit(y_class,y_pred,X_coalit):
    ' This function calculates the true/false positive/negative (confusion matrix) given a sensitive group '
    
    tn = sum((y_class < 0).astype(float) * np.array(y_pred < 0) * np.array(X_coalit == 1))
    tp = sum((y_class > 0).astype(float) * np.array(y_pred > 0) * np.array(X_coalit == 1))
    fn = sum((y_class < 0).astype(float) * np.array(y_pred > 0) * np.array(X_coalit == 1))
    fp = sum((y_class > 0).astype(float) * np.array(y_pred < 0) * np.array(X_coalit == 1))
    
    return tn,tp,fn,fp

def func_number_attr_names(X,features_to_encode):
    ' This function calculates the number of attributes and the list of attributes names '
    
    features_all = list() # All features and sub-features
    features_categorical = list() # All sub-features
    features_all_coalit = list() # Coalitions of features
    indices_categorical_features = list() # Indices of categorical features
    
    for ii in range(X.shape[1]):
        X_aux = X.loc[:,X.columns[ii]].to_frame()
        
        if len(set(features_to_encode).intersection({X.columns[ii]})) > 0:
            
            # One-hot-encoding and data transformation
            ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
            X_trans = ohe.fit_transform(X_aux).astype(float)
            features_encoded = ohe.get_feature_names_out(X_aux.columns) # Encoded features name
            indices_categorical_features.append(ii) # Indices of encoded features
            
            for jj in range(len(features_encoded)):
                
                features_all.append(features_encoded[jj])
                features_categorical.append(features_encoded[jj])
            
            if ii < (X.shape[1]-1):
                for qq in range(ii+1,X.shape[1]):
                    
                    X_aux2 = X.loc[:,X.columns[qq]].to_frame()
                    
                    if len(set(features_to_encode).intersection({X.columns[qq]})) > 0:
                        
                        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
                        X_trans = ohe.fit_transform(X_aux2).astype(float)
                        features_encoded2 = ohe.get_feature_names_out(X_aux2.columns)
                        
                        for aa in range(len(features_encoded)):
                            
                            for bb in range(len(features_encoded2)):
                                
                                features_all_coalit.append(features_encoded[aa]+','+features_encoded2[bb])
                
        else:
            features_all.append({X.columns[ii]})
    
        n_categ_ohe = len(features_categorical) # Number of categorical features after OHE
        n_coalit = len(features_all_coalit) # Number of categorical features after OHE
    
    return features_all,features_categorical,features_all_coalit,indices_categorical_features,n_categ_ohe,n_coalit

def func_kernel(Y, param_hsic):
    ' This function calculates the centered linear kernel of Y - Delta kernel is also available'
    n = len(Y)
    Y = np.reshape(np.array(Y), (n,1))
    H = np.eye(n) - (1/n)*np.ones((n,n))

    if(param_hsic == 'linear'):
      Hsic = Y @ Y.T #+ np.eye(n)  

    elif(param_hsic == 'delta'):
      A = np.array(Y @ Y.T)
      rows, cols = np.where(A == -1)
      A[rows, cols] = 0  
      Hsic = A + np.eye(n)
      
    else:
      Hsic = Y @ Y.T + np.eye(n)

    return H @ Hsic @ H

def func_kernel_matrix(X, param_hsic):
    ' This function calculates the centered linear kernel of Y - Delta kernel is also available'
    n,m = X.shape
    X = np.array(X)
    H = np.eye(n) - (1/n)*np.ones((n,n))

    if(param_hsic == 'linear'):
      Hsic = X @ X.T #+ np.eye(n)  

    elif(param_hsic == 'delta'):
      A = np.array(X @ X.T)
      rows, cols = np.where(A == -1)
      A[rows, cols] = 0  
      Hsic = A + np.eye(n)
      
    else:
      Hsic = X @ X.T + np.eye(n)

    return H @ Hsic @ H

def func_kernel_rbf(Y):
    ' This function calculates the centered RBF kernel of Y'
    n = len(Y)
    Y = np.reshape(np.array(Y), (n,1))
    H = np.eye(n) - (1/n)*np.ones((n,n))

    Hsic = rbf_kernel(Y)
    
    return H @ Hsic @ H

def func_kernel_rbf_matrix(X):
    ' This function calculates the centered RBF kernel of Y'
    n,m = X.shape
    X = np.array(X)
    H = np.eye(n) - (1/n)*np.ones((n,n))

    Hsic = rbf_kernel(X)
    
    return H @ Hsic @ H

def powerset(iterable,nAttr):
    '''Return the powerset of a set of m attributes
    powerset([1,2,..., m],m) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m) ... (1, ..., m)
    powerset([1,2,..., m],2) --> () (1,) (2,) (3,) ... (m,) (1,2) (1,3) ... (1,m) ... (m-1,m)
    '''
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(nAttr+1))

def tr_shap2game(nAttr):
    '''Return the transformation matrix from Shapley interaction indices, given a k_additive model, to game'''
    nBern = bernoulli(nAttr) #Números de Bernoulli
    k_add_numb = nParam_kAdd(nAttr,nAttr)

    coalit = np.zeros((k_add_numb,nAttr))

    for i,s in enumerate(powerset(range(nAttr),nAttr)):
        s = list(s)
        coalit[i,s] = 1

    matrix_shap2game = np.zeros((k_add_numb,k_add_numb))
    for i in range(coalit.shape[0]):
        for i2 in range(k_add_numb):
            aux2 = int(sum(coalit[i2,:]))
            aux3 = int(sum(coalit[i,:] * coalit[i2,:]))
            aux4 = 0
            for i3 in range(int(aux3+1)):
                aux4 += comb(aux3, i3) * nBern[aux2-i3]
            matrix_shap2game[i,i2] = aux4
    return matrix_shap2game

from scipy.special import bernoulli

def nParam_kAdd(kAdd,nAttr):
    '''Return the number of parameters in a k-additive model'''
    aux_numb = 1
    for ii in range(kAdd):
        aux_numb += comb(nAttr,ii+1)
    return aux_numb

def plot_hsic_fair(hsic_values,fair_measure,features_to_encode,fair_name,fair_name2):
    order_ind = np.argsort(-hsic_values).astype(int)
    order_hsic = hsic_values[order_ind]
    order_fair = fair_measure[order_ind]
    features_order = list()
    markers = ['s','o','+','x','h','v','<','>','p','d','*','.',',','^','1','2','3','4','8','P','H','X','D','|','_']
    for ii in range(len(hsic_values)):
        plt.plot(order_hsic[ii],order_fair[ii],marker=markers[ii], markersize=8)
        features_order.append(features_to_encode[order_ind[ii]])
    plt.legend(features_order, fontsize='11', ncol=1, loc='upper left')
    plt.xlabel(fair_name, fontsize='12')
    plt.ylabel(fair_name2, fontsize='12')
    plt.show()

def gradientbars(bars):
    grad = np.atleast_2d(np.linspace(0,1,256))
    ax = bars[0].axes
    lim = ax.get_xlim()+ax.get_ylim()
    for bar in bars:
        bar.set_zorder(1)
        bar.set_facecolor('none')
        x,y = bar.get_xy()
        w, h = bar.get_width(), bar.get_height()
        ax.imshow(grad, extent=[x+w, x, y, y+h], aspect='auto', zorder=1)
    ax.axis(lim)
   
def func_hsic_values(hsic_values,features_names,measure):
    order_ind = np.argsort(hsic_values).astype(int)
    order = hsic_values[order_ind]
    names = list()
    for ii in range(len(hsic_values)):
        names.append(features_names[order_ind[ii]])
    bar = plt.barh(names,order)
    plt.yticks(np.arange(len(hsic_values)), names, rotation=0, fontsize='11')
    plt.xlabel(measure, fontsize='13')
    plt.ylabel('Features', fontsize='13')
    gradientbars(bar)
    plt.show()
    
