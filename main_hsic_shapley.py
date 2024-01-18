"""
This script proposes an approach to detect relevant and disparity prone feaures before training

We consider the following datasets in our analysis:
- Adult Income: https://archive.ics.uci.edu/ml/datasets/adult
- COMPAS Recidivism risk: https://www.propublica.org/article/machine-bias-risk-assessments-in-criminal-sentencing
- LSAC: http://www.seaphe.org/databases.php
- Rice: https://archive.ics.uci.edu/dataset/545/rice+cammeo+and+osmancik
- Banknotes: https://archive.ics.uci.edu/dataset/267/banknote+authentication
- Red Wine Quality: https://archive.ics.uci.edu/dataset/186/wine+quality
- Diabetes Pima: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
- Raisin: https://archive.ics.uci.edu/dataset/850/raisin
"""

''' Importing packages '''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
import mod_SensDetec_hsic_shapley
from sklearn.model_selection import train_test_split
import seaborn as sns
import math


''' Importing dataset '''
# data_imp is the dataset to import (random, adult, compas, lsac_new, rice, banknotes, redwine, diabetesPima, raisin)
# n_samp is the number of random samples to consider when calculating HSIC (set 0 if use all samples)
# X and y are the full dataset; X and y are the subsamples to calculate HSIC (if n_samp ~= 0)
data_imp = 'compas'
X, y = mod_SensDetec_hsic_shapley.func_read_data(data_imp)


''' Splitting train and test '''
indices_split_aux = np.arange(X.shape[0])
X_train, X_test, y_train, y_test, indices_split_train, indices_split_test = train_test_split(X, y, indices_split_aux, test_size=0.2)

''' Parameters '''
nSamp_train = X_train.shape[0] # Number of samples in sensitive group A 
nSamp_test = X_test.shape[0] # Number of samples in sensitive group A 
features_names = X.columns # Features names
features_to_encode = X.columns[X.dtypes==object].tolist() # Categorical features names
n_categ = len(features_to_encode)
n_feat = X.shape[1]

# Categorical features and number of features after OHE
features_all,features_categorical,features_all_coalit,indices_categorical_features,n_categ_ohe,n_coalit = mod_SensDetec_hsic_shapley.func_number_attr_names(X,features_to_encode)
indices_numerical_features = np.setdiff1d(np.arange(n_feat),indices_categorical_features).tolist()

''' Pipeline of the classifier '''
seed=50
'''
model = RandomForestClassifier(
                      min_samples_leaf=50,
                      n_estimators=150,
                      bootstrap=True,
                      oob_score=True,
                      n_jobs=-1,
                      random_state=seed,
                      max_features='auto')
'''        
model = MLPClassifier(max_iter=1000,random_state=seed)

''' HSIC / NOCCO parameters '''
# Predefined parameters
param_hsic = 'linear' # HSIC kernel parameter
param_nocco = 10**(-6) # NOCCO regularizer parameter

''' Predefined matrices and vectors '''
# Payoffs (Hsic values)
hsic_values = np.zeros((2**n_feat,))

# Payoffs (Performance measures)
tp_shapley = np.zeros((2**n_feat,))
tn_shapley = np.zeros((2**n_feat,))
fp_shapley = np.zeros((2**n_feat,))
fn_shapley = np.zeros((2**n_feat,))

tp_shapley_train = np.zeros((2**n_feat,))
tn_shapley_train = np.zeros((2**n_feat,))
fp_shapley_train = np.zeros((2**n_feat,))
fn_shapley_train = np.zeros((2**n_feat,))

# True/false positive/negative vectors for test and train and for sensitive groups
tp_g, tp_g_train = np.zeros((n_categ_ohe,)), np.zeros((n_categ_ohe,))
tn_g, tn_g_train = np.zeros((n_categ_ohe,)), np.zeros((n_categ_ohe,))
fp_g, fp_g_train = np.zeros((n_categ_ohe,)), np.zeros((n_categ_ohe,))
fn_g, fn_g_train = np.zeros((n_categ_ohe,)), np.zeros((n_categ_ohe,))

# Fairness metrics for test and train
ov_ac = np.zeros((len(features_to_encode,)))
ov_ac_train = np.zeros((len(features_to_encode,)))

# Matrices for coalition analysis
tp_coalit = np.zeros((n_coalit,))
tn_coalit = np.zeros((n_coalit,))
fp_coalit = np.zeros((n_coalit,))
fn_coalit = np.zeros((n_coalit,))

tp_coalit_train = np.zeros((n_coalit,))
tn_coalit_train = np.zeros((n_coalit,))
fp_coalit_train = np.zeros((n_coalit,))
fn_coalit_train = np.zeros((n_coalit,))

''' HSIC Shapley values analysis '''
# HSIC for label vector
#hsic_aux_y = mod_SensDetec_hsic_shapley.func_kernel(y_train, param_hsic)
hsic_aux_y = mod_SensDetec_hsic_shapley.func_kernel_rbf(y_train)
hsic_norm = hsic_aux_y @ np.linalg.inv(hsic_aux_y + nSamp_train*param_nocco*np.eye(nSamp_train))

for i,s in enumerate(mod_SensDetec_hsic_shapley.powerset(range(n_feat),n_feat)):
    if len(s)>0:
        s = list(s)
        XX = X_train.iloc[:,s]
        XX_test = X_test.iloc[:,s]
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        XX_trans = np.empty(shape=[nSamp_train,1])
        XX_test_trans = np.empty([nSamp_test,1])
        
        for ii in range(len(s)):
            if len(set(features_to_encode).intersection({XX.columns[ii]})) > 0:
                
                XX_trans = np.append(XX_trans,np.array(ohe.fit_transform(XX.iloc[:,ii].to_frame())),axis=1)
                XX_test_trans = np.append(XX_test_trans,np.array(ohe.fit_transform(XX_test.iloc[:,ii].to_frame())),axis=1)
                
            else:
                XX_trans = np.append(XX_trans,np.array(XX.iloc[:,ii]).reshape(nSamp_train,1),axis=1)
                XX_test_trans = np.append(XX_test_trans,np.array(XX_test.iloc[:,ii]).reshape(nSamp_test,1),axis=1)
        
                
        XX_trans = np.delete(XX_trans, 0, 1)
        XX_test_trans = np.delete(XX_test_trans, 0, 1)
        
        XX_trans = (XX_trans - np.mean(XX_trans, axis=0))/np.std(XX_trans, axis=0)
        XX_test_trans = (XX_test_trans - np.mean(XX_test_trans, axis=0))/np.std(XX_test_trans, axis=0)
        
        hsic_aux = mod_SensDetec_hsic_shapley.func_kernel_rbf_matrix(XX_trans)
        hsic_values[i] = np.trace((hsic_aux @ np.linalg.inv(hsic_aux + nSamp_train*param_nocco*np.eye(nSamp_train))) @ hsic_norm)
        
        
        
        #col_trans = make_column_transformer(
        #                        (OneHotEncoder(),features_to_encode_subset),
        #                        remainder = "passthrough"
        #                        )
        
        #pipe = make_pipeline(col_trans, model)
        #pipe.fit(XX, y_train)
        #y_pred = pipe.predict(XX_test) # Predictions for test data 
        
        pipe = make_pipeline(model)
        pipe.fit(XX_trans, y_train)
        y_pred = pipe.predict(XX_test_trans) # Predictions for test data 
           
        tn_shapley[i] = sum((y_pred < 0) * (y_test < 0))
        tp_shapley[i] = sum((y_pred > 0) * (y_test > 0))
        fn_shapley[i] = sum((y_pred < 0) * (y_test > 0))
        fp_shapley[i] = sum((y_pred > 0) * (y_test < 0))
        
        y_pred_train = pipe.predict(XX_trans) # Predictions for test data 
           
        tn_shapley_train[i] = sum((y_pred_train < 0) * (y_train < 0))
        tp_shapley_train[i] = sum((y_pred_train > 0) * (y_train > 0))
        fn_shapley_train[i] = sum((y_pred_train < 0) * (y_train > 0))
        fp_shapley_train[i] = sum((y_pred_train > 0) * (y_train < 0))
    
        print(i,'/',2**n_feat)   
        print(XX_trans.shape[1],'/',XX_test_trans.shape[1])
    
transf_matrix = np.linalg.inv(mod_SensDetec_hsic_shapley.tr_shap2game(n_feat)) # Transformation matrix from game to Shapley domain
shapley_values = transf_matrix @ hsic_values

# Fairness analysis based on categorical features (without spliting categories)
tpr_shapley, fpr_shapley = tp_shapley/(tp_shapley+fn_shapley), fp_shapley/(fp_shapley+tn_shapley)
tpr_shapley[np.isnan(tpr_shapley)], fpr_shapley[np.isnan(fpr_shapley)] = 0, 0
accur_shapley = (tp_shapley + tn_shapley)/(tp_shapley + tn_shapley + fp_shapley + fn_shapley)
accur_shapley[np.isnan(accur_shapley)] = 0

shapley_values_tpr = transf_matrix @ tpr_shapley
shapley_values_fpr = transf_matrix @ fpr_shapley
shapley_values_accur = transf_matrix @ accur_shapley


tpr_shapley_train, fpr_shapley_train = tp_shapley_train/(tp_shapley_train+fn_shapley_train), fp_shapley_train/(fp_shapley_train+tn_shapley_train)
tpr_shapley_train[np.isnan(tpr_shapley_train)], fpr_shapley_train[np.isnan(fpr_shapley_train)] = 0, 0
accur_shapley_train = (tp_shapley_train + tn_shapley_train)/(tp_shapley_train + tn_shapley_train + fp_shapley_train + fn_shapley_train)
accur_shapley_train[np.isnan(accur_shapley_train)] = 0

shapley_values_tpr_train = transf_matrix @ tpr_shapley_train
shapley_values_fpr_train = transf_matrix @ fpr_shapley_train
shapley_values_accur_train = transf_matrix @ accur_shapley_train

''' Plots '''
if data_imp == 'random':
    mod_SensDetec_hsic_shapley.func_hsic_values(shapley_values[1:n_feat+1],features_names,'NOCCO Shapley value')
    mod_SensDetec_hsic_shapley.func_hsic_values(hsic_values[1:n_feat+1],features_names, 'Dependence measure (NOCCO)')
    
    # Interaction indices (all features)
    combinat = np.zeros((int(n_feat*(n_feat-1)/2),2))
    count = 0
    indices = np.zeros((n_feat,n_feat))
    mix_names = list()
    for ii in range(n_feat-1):
        for jj in range(ii+1,n_feat):
            combinat[count,0], combinat[count,1] = ii, jj
            indices[ii,jj] = shapley_values[n_feat+count+1]
            count += 1
        mix_names.append(features_names[ii]) # RC means Random Classifier
    mix_names.append(features_names[-1])
    indices = indices + indices.T
    
    ax = sns.heatmap(
        indices, 
        vmin=np.min(indices), vmax=np.max(indices), center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        xticklabels=mix_names,
        yticklabels=mix_names
    )
    plt.figure(figsize = (25,5))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    plt.show()
    
    # For the random dataset
    data_save = [hsic_values,shapley_values,accur_shapley,tpr_shapley,fpr_shapley,shapley_values_tpr,shapley_values_fpr,shapley_values_accur,accur_shapley_train,tpr_shapley_train,fpr_shapley_train,shapley_values_tpr_train,shapley_values_fpr_train,shapley_values_accur_train,n_feat,features_names]
    np.save('results_SensDetec_hsic_shapley_test.npy', data_save, allow_pickle=True)

    
    exit()
else: 
    # Shapley values
    mod_SensDetec_hsic_shapley.func_hsic_values(shapley_values[1:n_feat+1],features_names,'NOCCO Shapley value ($\phi^{NOCCO}$)')
    mod_SensDetec_hsic_shapley.func_hsic_values(hsic_values[1:n_feat+1],features_names, 'Dependence measure (NOCCO)')
    mod_SensDetec_hsic_shapley.func_hsic_values(shapley_values_accur[1:n_feat+1],features_names, 'Overal accuracy Shapley value ($\phi^{OA}$)')
    mod_SensDetec_hsic_shapley.func_hsic_values(shapley_values_accur_train[1:n_feat+1],features_names, 'Overal accuracy Shapley value ($\phi^{OA}$)')
    
    # Shapley values x Overall accuracy
    mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values[1:n_feat+1],shapley_values_accur[1:n_feat+1],features_names,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overal accuracy Shapley value ($\phi^{OA}$)')
    mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values[1:n_feat+1],shapley_values_accur_train[1:n_feat+1],features_names,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overal accuracy Shapley value ($\phi^{OA}$)')
    
    # Interaction indices (all features)
    combinat = np.zeros((int(n_feat*(n_feat-1)/2),2))
    count = 0
    indices = np.zeros((n_feat,n_feat))
    mix_names = list()
    for ii in range(n_feat-1):
        for jj in range(ii+1,n_feat):
            combinat[count,0], combinat[count,1] = ii, jj
            indices[ii,jj] = shapley_values[n_feat+count+1]
            count += 1
        mix_names.append(features_names[ii]) # RC means Random Classifier
    mix_names.append(features_names[-1])
    indices = indices + indices.T
    
    ax = sns.heatmap(
        indices, 
        vmin=np.min(indices), vmax=np.max(indices), center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        xticklabels=mix_names,
        yticklabels=mix_names
    )
    plt.figure(figsize = (25,5))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    plt.show()
    
    # Interaction indices (categorical features)
    indices_categ = indices[indices_categorical_features,:]
    indices_categ = indices_categ[:,indices_categorical_features]
    ax = sns.heatmap(
        indices_categ, 
        vmin=np.min(indices_categ), vmax=np.max(indices_categ), center=0,
        cmap=sns.diverging_palette(20, 220, n=200),
        square=True,
        xticklabels=features_to_encode,
        yticklabels=features_to_encode
    )
    plt.figure(figsize = (25,5))
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    ax.set_yticklabels(
        ax.get_yticklabels(),
        rotation=0,
        horizontalalignment='right', 
        fontsize = 13.0
    )
    plt.show()


''' Classification analysis - Performance and fariness measures '''
sensitive = mod_SensDetec_hsic_shapley.func_sensitive_indices(data_imp)

col_trans = make_column_transformer(
                        (OneHotEncoder(),features_to_encode),
                        remainder = "passthrough"
                        )
pipe = make_pipeline(col_trans, model)
pipe.fit(X_train, y_train) # Classifier pipeline
y_class = pipe.predict(X_test) # Predictions for test data    
y_class_train = pipe.predict(X_train) # Predictions for training data

# True/false positive/negative for test and train
tn,tp,fn,fp = mod_SensDetec_hsic_shapley.func_confusion_matrix(y_class,y_test)
tn_train,tp_train,fn_train,fp_train = mod_SensDetec_hsic_shapley.func_confusion_matrix(y_class_train,y_train)


cont, cont2 = 0, 0 # Auxiliar variables

for ii in range(len(features_to_encode)):
    
    # One-hot-encoding and data transformation
    features_to_encode_aux = [features_to_encode[ii]]
    X_aux = X_test[features_to_encode_aux]
    X_aux_train = X_train[features_to_encode_aux]
    ohe = OneHotEncoder(handle_unknown='ignore')
    X_trans = ohe.fit_transform(X_aux).toarray() # Encoded data for test
    X_trans_train = ohe.fit_transform(X_aux_train).toarray()  # Encoded data for train
    features_encoded = ohe.get_feature_names_out(X_aux.columns) # Get the name of encoded categories
    
    
    cont_aux = np.copy(cont) # Auxiliar variable
    
    for jj in range(X_trans.shape[1]):
        
        # True/false positive/negative for test and train
        tn_g[cont],tp_g[cont],fn_g[cont],fp_g[cont] = mod_SensDetec_hsic_shapley.func_confusion_matrix_groups(y_class,y_test,X_aux,features_categorical,features_to_encode_aux,cont)
        tn_g_train[cont],tp_g_train[cont],fn_g_train[cont],fp_g_train[cont] = mod_SensDetec_hsic_shapley.func_confusion_matrix_groups(y_class_train,y_train,X_aux_train,features_categorical,features_to_encode_aux,cont)
        
        cont += 1
    
    # Fairness analysis based on categorical features (without spliting categories)
    tpr_groups, fpr_groups = tp_g[cont_aux:cont]/(tp_g[cont_aux:cont]+fn_g[cont_aux:cont]), fp_g[cont_aux:cont]/(fp_g[cont_aux:cont]+tn_g[cont_aux:cont])
    tpr_groups[np.isnan(tpr_groups)], fpr_groups[np.isnan(fpr_groups)] = 0, 0
    accur_groups = (tp_g[cont_aux:cont] + tn_g[cont_aux:cont])/(tp_g[cont_aux:cont] + tn_g[cont_aux:cont] + fp_g[cont_aux:cont] + fn_g[cont_aux:cont])
    accur_groups[np.isnan(accur_groups)] = 0
    
    tpr_groups_train, fpr_groups_train = tp_g_train[cont_aux:cont]/(tp_g_train[cont_aux:cont]+fn_g_train[cont_aux:cont]), fp_g_train[cont_aux:cont]/(fp_g_train[cont_aux:cont]+tn_g_train[cont_aux:cont])
    tpr_groups_train[np.isnan(tpr_groups_train)], fpr_groups_train[np.isnan(fpr_groups_train)] = 0, 0
    accur_groups_train = (tp_g_train[cont_aux:cont] + tn_g_train[cont_aux:cont])/(tp_g_train[cont_aux:cont] + tn_g_train[cont_aux:cont] + fp_g_train[cont_aux:cont] + fn_g_train[cont_aux:cont])
    accur_groups_train[np.isnan(accur_groups_train)] = 0
    
    for jj in range(X_trans.shape[1]-1):
        
        for ll in range(jj+1,X_trans.shape[1]):
        
            # Fairness metrics
            ov_ac[ii] +=  (1/math.comb(X_trans.shape[1],2))*np.abs(accur_groups[jj] - accur_groups[ll]) # Equal overall accuracy
            ov_ac_train[ii] +=  (1/math.comb(X_trans.shape[1],2))*np.abs(accur_groups_train[jj] - accur_groups_train[ll])
    
    # Fairness analysis based on coalition of subfeatures
    for qq in range(ii+1,len(features_to_encode)):
        features_to_encode_aux2 = [features_to_encode[qq]]
        X_aux2 = X_test[features_to_encode_aux2]
        X_aux2_train = X_train[features_to_encode_aux2]
        
        ohe = OneHotEncoder(handle_unknown='ignore', sparse=False)
        X_trans2 = ohe.fit_transform(X_aux2).astype(float)
        X_trans2_train = ohe.fit_transform(X_aux2_train).astype(float)
        features_encoded2 = ohe.get_feature_names_out(X_aux2.columns)
        
        for aa in range(X_trans.shape[1]):
                
            for bb in range(X_trans2.shape[1]):
                    
                X_coalit = X_trans[:,aa] * X_trans2[:,bb]
                X_coalit = X_coalit.reshape(nSamp_test,)
                
                X_coalit_train = X_trans_train[:,aa] * X_trans2_train[:,bb]
                X_coalit_train = X_coalit_train.reshape(nSamp_train,)
                    
                # Fariness analysis
                tn_coalit[cont2],tp_coalit[cont2],fn_coalit[cont2],fp_coalit[cont2] = mod_SensDetec_hsic_shapley.func_confusion_matrix_coalit(y_class,y_test,X_coalit)
                tn_coalit_train[cont2],tp_coalit_train[cont2],fn_coalit_train[cont2],fp_coalit_train[cont2] = mod_SensDetec_hsic_shapley.func_confusion_matrix_coalit(y_class_train,y_train,X_coalit_train)
                    
                cont2 += 1
    print(ii)

tpr_coalit, fpr_coalit = tp_coalit/(tp_coalit+fn_coalit), fp_coalit/(fp_coalit+tn_coalit)
tpr_coalit[np.isnan(tpr_coalit)], fpr_coalit[np.isnan(fpr_coalit)] = 0, 0
accur_coalit = (tp_coalit + tn_coalit)/(tp_coalit + tn_coalit + fp_coalit + fn_coalit)
accur_coalit[np.isnan(accur_coalit)] = 0

tpr_coalit_train, fpr_coalit_train = tp_coalit_train/(tp_coalit_train+fn_coalit_train), fp_coalit_train/(fp_coalit_train+tn_coalit_train)
tpr_coalit_train[np.isnan(tpr_coalit_train)], fpr_coalit_train[np.isnan(fpr_coalit_train)] = 0, 0
accur_coalit_train = (tp_coalit_train + tn_coalit_train)/(tp_coalit_train + tn_coalit_train + fp_coalit_train + fn_coalit_train)
accur_coalit_train[np.isnan(accur_coalit_train)] = 0

''' Plots '''
# Shapley values x Fairness measure
shapley_values_aux = shapley_values[1:n_feat+1]
mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values_aux[indices_categorical_features],ov_ac,features_to_encode,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overall accuracy equality (OAE)')
mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values_aux[indices_categorical_features],ov_ac_train,features_to_encode,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overall accuracy equality (OAE)')

indices_sensitive = []
features_sensitive = []
for ii in range(len(sensitive)):
    indices_sensitive.append(np.where(np.array(sensitive)[ii]==np.array(indices_categorical_features)))
    features_sensitive.append(features_to_encode[int(np.where(np.array(sensitive)[ii]==np.array(indices_categorical_features))[0])])
    
mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values_aux[np.array(sensitive)],ov_ac[indices_sensitive],features_sensitive,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overall accuracy equality (OAE)')
mod_SensDetec_hsic_shapley.plot_hsic_fair(shapley_values_aux[np.array(sensitive)],ov_ac_train[indices_sensitive],features_sensitive,'NOCCO Shapley value ($\phi^{NOCCO}$)','Overall accuracy equality (OAE)')


''' Save '''
# Without fairness analysis
#data_save = [hsic_values,shapley_values,accur_shapley,tpr_shapley,fpr_shapley,shapley_values_tpr,shapley_values_fpr,shapley_values_accur,accur_shapley_train,tpr_shapley_train,fpr_shapley_train,shapley_values_tpr_train,shapley_values_fpr_train,shapley_values_accur_train,n_feat,features_names]
#np.save('results_SensDetec_hsic_shapley_test.npy', data_save, allow_pickle=True)
#hsic_values,shapley_values,accur_shapley,tpr_shapley,fpr_shapley,shapley_values_tpr,shapley_values_fpr,shapley_values_accur,accur_shapley_train,tpr_shapley_train,fpr_shapley_train,shapley_values_tpr_train,shapley_values_fpr_train,shapley_values_accur_train,n_feat,features_names = np.load('results_SensDetec_hsic_shapley_test.npy', allow_pickle=True)

# With fairness analysis
data_save = [hsic_values,shapley_values,accur_shapley,tpr_shapley,fpr_shapley,shapley_values_tpr,shapley_values_fpr,shapley_values_accur,accur_shapley_train,tpr_shapley_train,fpr_shapley_train,shapley_values_tpr_train,shapley_values_fpr_train,shapley_values_accur_train,n_feat,features_names,indices_categorical_features,features_to_encode,tpr_coalit,fpr_coalit,accur_coalit,tpr_coalit_train,fpr_coalit_train,accur_coalit_train,shapley_values_aux,ov_ac,ov_ac_train,features_all_coalit,sensitive]
np.save('results_SensDetec_hsic_shapley_test.npy', data_save, allow_pickle=True)
hsic_values,shapley_values,accur_shapley,tpr_shapley,fpr_shapley,shapley_values_tpr,shapley_values_fpr,shapley_values_accur,accur_shapley_train,tpr_shapley_train,fpr_shapley_train,shapley_values_tpr_train,shapley_values_fpr_train,shapley_values_accur_train,n_feat,features_names,indices_categorical_features,features_to_encode,tpr_coalit,fpr_coalit,accur_coalit,tpr_coalit_train,fpr_coalit_train,accur_coalit_train,shapley_values_aux,ov_ac,ov_ac_train,features_all_coalit,sensitive = np.load('results_SensDetec_hsic_shapley_test.npy', allow_pickle=True)



