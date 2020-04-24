
import numpy as np
import pickle
from flask import Flask, request, jsonify, render_template

app=Flask(__name__)
model=pickle.load(open('deploy2.pkl','rb'))

@app.route('/')
def hellow():
    return render_template('index1.html')

@app.route('/predicts',methods=['POST'])
def predicts():
    features=[float(i) for i in request.form.values()]
    features=[np.array(features)]
    a=model.predict(features)
    output = round(a[0], 2)
    return render_template('index1.html',prediction_text='the flower is {}'.format(output))

if  __name__ =='__main__':
   app.run(debug=True, use_reloader=False)
    