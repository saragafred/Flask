# Dependencies
#from os import system
#from flask import Flask, request, jsonify
#import json
#import random
#import traceback
#import pandas as pd
#import pickle
from flask import Flask, request, jsonify
import pickle
import pandas as pd
import lightgbm


#Load Dataframe
###---------- load data -------- 
def load_all_data():
    
    data = pd.read_csv("./data/train_set_echantillon.csv")

    return data




#@app.route('/post/<int:id>')
#def show_post(id):
    # Shows the post with given id.
    #return f'This post has the id {id}'


#@app.route("/credit")
#def home_view():
        
        #data = load_all_data()
        #model = pickle.load(open("./modelisation/classifier_lgbm_model.sav", 'rb'))
        
        #return "Hello, World!"

app = Flask(__name__)
@app.route('/credit/<int:id_client>', methods=['GET'])
def credit(id_client):

    data = load_all_data()
    model = pickle.load(open("./modelisation/classifier_lgbm_model.sav", 'rb'))
    ID = int(id_client)
    X = data[data['SK_ID_CURR'] == ID]
    X = X.drop(['SK_ID_CURR', 'TARGET'], axis=1)
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)

    dict_final = {
        'prediction' : int(y_pred),
        'proba' : float(y_proba[0][0])
        }

    print('Nouvelle Prédiction : \n', dict_final)

    return jsonify(dict_final)
   

if __name__ == "__main__":
    app.run(debug=True)



