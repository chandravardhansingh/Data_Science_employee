import pandas as pd
import numpy as np
data= pd.read_csv("aug_train.csv")

y = data.iloc[:,-1]
x = data.iloc[:,2:-1]

features =  ['12', 'Male', 'Has relevent experience', 'no_enrollment', 'Primary School', 'STEM', '34', '50-99', 'Pvt Ltd', '1', '34']
column_name = [ 'city_development_index', 'gender', 'relevent_experience',
       'enrolled_university', 'education_level', 'major_discipline',
       'experience', 'company_size', 'company_type', 'last_new_job',
       'training_hours']
test_row = pd.DataFrame([features],columns=column_name)


x.experience = x.experience.replace({"<1":0,">20":21})
x.experience = pd.to_numeric(x.experience)

if int(test_row['experience'][0]) > 20:
    test_row['experience'] = 21

from sklearn.impute import SimpleImputer

col1 = ['experience','enrolled_university','last_new_job','education_level']
col2 = ['major_discipline','gender','company_size','company_type']


def modeimpute(df,imputer,cols):
    new_df = pd.DataFrame(imputer.fit_transform(df[cols]),columns=cols)
    df.drop(cols,axis=1,inplace=True)
    return df.join(new_df)
    
si1 = SimpleImputer(strategy='most_frequent')
x = modeimpute(x,si1,col1)

for col in col2:
    x[col+"_missing"] = np.where(x[col].isnull(),1,0)

si2 = SimpleImputer(strategy='most_frequent')
x = modeimpute(x,si2,col2)

for col in col2:
    test_row[col+"_missing"] = np.where(test_row[col].isnull(),1,0)
from sklearn.preprocessing import OneHotEncoder
cols = ['gender', 'relevent_experience',
       'enrolled_university', 'education_level', 'major_discipline',
        'company_size', 'company_type', 'last_new_job']

OH_encoder = OneHotEncoder(drop='first', sparse=False)
OH_col = pd.DataFrame(OH_encoder.fit_transform(x[cols]), columns=OH_encoder.get_feature_names())
OH_col.index = x.index
x = x.drop(cols, axis=1)
x = x.join(OH_col)

OH_col2 = pd.DataFrame(OH_encoder.transform(test_row[cols]), columns=OH_encoder.get_feature_names())
OH_col2.index = test_row.index
test_row = test_row.drop(cols, axis=1)
test_row = test_row.join(OH_col)

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
x=pd.DataFrame(scaler.fit_transform(x),columns=x.columns)

test_row = pd.DataFrame(scaler.fit_transform(test_row),columns=test_row.columns)

#####  Training the model  #####
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
# dtc = DecisionTreeClassifier()
# dtc.fit(x,y)

import pickle
# pickle.dump(dtc,open("dtcmodel.pkl", 'wb'))

model = pickle.load(open('dtcmodel.pkl','rb'))
predicted = model.predict(test_row)
predicted = int(predicted[0])