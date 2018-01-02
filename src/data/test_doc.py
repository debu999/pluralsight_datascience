import os
ml_script_fl = os.path.join(os.pardir, "src", "models", "machine_learning_api.py")
print(ml_script_fl)
print(os.getcwd())


%%writefile $ml_script_fl

from flask import Flask, request
import pandas as pd
import numpy as np
import os, sys, json, pickle

app = Flask(__name__)

"""Load Model and Scalar files"""
model_file_path = os.path.join(os.path.pardir, os.path.pardir, 'models', 'lr_model.pkl')
scalar_file_path = os.path.join(os.path.pardir, os.path.pardir, 'models', 'lr_scalar.pkl')
mdl_fl_pkl = open(model_file_path,"rb")
scl_fl_pkl = open(scalar_file_path,"rb")

model = pickle.load(mdl_fl_pkl)
scalar = pickle.load(scl_fl_pkl)

columns = [u'Age', u'Fare', u'FamilySize', u'IsMother', u'IsMale',
           u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', u'Deck_E',
           u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1', u'Pclass_2',
           u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss',
           u'Title_Mr', u'Title_Mrs', u'Title_Officer', u'Title_Sir',
           u'Fare_Bin_very_low', u'Fare_Bin_low', u'Fare_Bin_medium',
           u'Fare_Bin_high', u'Embarked_C', u'Embarked_Q', u'Embarked_S',
           u'Agestate_Adults', u'Agestate_Child']

@app.route('/api', methods=['POST'])
def make_predictions():
    data = json.dumps(request.get_json(force=True))
    df = pd.read_json(data)
    passenger_ids = df.PassengerId.ravel()
    survivals = df.Survived.ravel()
    X = df[columns].as_matrix().astype('float')
    Xsc = scalar.transform(X)
    p = model.predict(Xsc)
    df_response = pd.DataFrame({"PassengerId":passenger_ids, "Predicted":p, "Actuals":survivals})
    return df_response.to_json()
    
    name = data['name']
    return "Hello to API World : {0}".format(name)

if __name__ == "__main__":
    app.run(port=8293, debug=True)


import json, requests
url = "http://127.0.0.1:8292/api"
data = json.dumps({"name":"Debabrata"})
r = requests.post(url,data)
r.text

! ls -lrt ..\src\models\

### API Invocation 

import os
import pandas as pd
processed_data_path = os.path.join(os.path.pardir, "data", "processed")
tr_fl_pt = os.path.join(processed_data_path, "train.csv")
train_df = pd.read_csv(tr_fl_pt)
sur_pass = train_df[train_df['Survived']==1][:10]



sur_pass

import requests
def make_api_request(data):
    url = "http://127.0.0.1:8293/api"
    r = requests.post(url,data)
    print(r.status_code)
    return r.json()

from pprint import pprint as pp
pp(make_api_request(sur_pass.to_json()))

res = make_api_request(train_df.to_json())
df_res = pd.read_json(json.dumps(res))
df_res.head()

import numpy as np
np.mean(df_res.Actuals == df_res.Predicted)

from flask import Flask, request
import pandas as pd
import numpy as np
import os, sys, json, pickle
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split


"""Load Model and Scalar files"""
model_file_path = os.path.join( os.path.pardir, 'models', 'lr_model.pkl')
scalar_file_path = os.path.join( os.path.pardir, 'models', 'lr_scalar.pkl')
mdl_fl_pkl = open(model_file_path,"rb")
scl_fl_pkl = open(scalar_file_path,"rb")

model = pickle.load(mdl_fl_pkl)
scalar = pickle.load(scl_fl_pkl)

columns = [u'Age', u'Fare', u'FamilySize', u'IsMother', u'IsMale',
           u'Deck_A', u'Deck_B', u'Deck_C', u'Deck_D', u'Deck_E',
           u'Deck_F', u'Deck_G', u'Deck_Z', u'Pclass_1', u'Pclass_2',
           u'Pclass_3', u'Title_Lady', u'Title_Master', u'Title_Miss',
           u'Title_Mr', u'Title_Mrs', u'Title_Officer', u'Title_Sir',
           u'Fare_Bin_very_low', u'Fare_Bin_low', u'Fare_Bin_medium',
           u'Fare_Bin_high', u'Embarked_C', u'Embarked_Q', u'Embarked_S',
           u'Agestate_Adults', u'Agestate_Child']

import os
import pandas as pd
processed_data_path = os.path.join(os.path.pardir, "data", "processed")
tr_fl_pt = os.path.join(processed_data_path, "train.csv")
train_df = pd.read_csv(tr_fl_pt)
sur_pass = train_df[train_df['Survived']==1][:10]
y=train_df['Survived']
X = train_df[columns].as_matrix().astype('float')
Xsc = scalar.transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)

import os
import pandas as pd
processed_data_path = os.path.join(os.path.pardir, "data", "processed")
tr_fl_pt = os.path.join(processed_data_path, "train.csv")
train_df = pd.read_csv(tr_fl_pt)
sur_pass = train_df[train_df['Survived']==1][:10]
y=train_df['Survived']
X = train_df[columns].as_matrix().astype('float')
Xsc = scalar.transform(X)
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=0.2, random_state=0)
nn = MLPClassifier(activation='logistic', solver='sgd', hidden_layer_sizes=(100,50),random_state=1)
nn.fit(Xtrain, ytrain)
pred=nn.predict(Xtest)
z=pd.DataFrame(pred)
e=pd.concat([z,ytest],axis=1)
e.head()

def make_predictions():
    data = train_df.to_json()
    df = pd.read_json(data)
    passenger_ids = df.PassengerId.ravel()
    survivals = df.Survived.ravel()
    X = df[columns].as_matrix().astype('float')
    Xsc = scalar.transform(X)
    p = model.predict(Xsc)
    df_response = pd.DataFrame({"PassengerId":passenger_ids, "Predicted":p, "Actuals":survivals})
    return df_response



x=make_predictions()
x[["Actuals","Predicted","PassengerId"]]
