import pickle
from flask import Flask,request,jsonify
import json

def predict(usrinp):
 inp=[usrinp]
 loaded_model = pickle.load(open('finalized_model.sav', 'rb'))
 y_pred=loaded_model.predict(inp)
 if y_pred[0]=='spam':
     return 1
 else:
     return 0
app = Flask(__name__)

@app.route('/', methods=['GET'])
def getspam():
    text=request.args.get('text')
    spam=predict(text)
    return str(spam)
   
if __name__ == '__main__':
    app.run(debug=True)
#predict("Your mobile number has won cash award")
#print(predict("Your mobile number has won cash award"))