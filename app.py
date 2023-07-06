import pickle

from flask import Flask, render_template, request
from sklearn.preprocessing import StandardScaler

# from ml_model import X_train

app=Flask(__name__)

#Deserialise / Depickle
clf=pickle.load(open('model.pkl','rb'))


@app.route("/")
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])
    features=[int(x) for x in request.form.values()] #Maintain input same as the date while training
    with open('scaler.pkl','rb') as file:
        sst=pickle.load(file)
        output=clf.predict(sst.transform([features]))
        print(output)
    if output[0]==0:
        return render_template("index.html",pred="The Person will not purchase the SUV")
    else:
        return render_template("index.html",pred="The Person will purchase the SUV")

if __name__ == "__main__":
    app.run(debug=True) #create a flask local server