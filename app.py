import pickle
from flask import Flask, render_template, request

app=Flask(__name__)

#Deserialize / depickle - loading the model
clf=pickle.load(open('model.pkl','rb'))

#jinja2 template - Template engine - will select the templates from the templates folder
@app.route("/") #decorators - when user access the root URL - func gets invoked
def hello():
    return render_template("index.html")

@app.route('/predict',methods=['POST','GET'])
def predict_class():
    print([x for x in request.form.values()])#POST method is working fine - communication created success
    features=[int(x) for x in request.form.values()] #maintain the inp same as the data u trained the model
    #label encode, normalize - 2 conflicts
    with open('sst.pkl', 'rb') as file:
        sst = pickle.load(file)

    output = clf.predict(sst.transform([features]))

    #output=clf.predict([features])

    print(output) #[1]
    if output[0]==0: #GET METHOD
        return render_template("index.html",pred="The person will not purchase the SUV")

    else:
        return render_template("index.html",pred="The person will purchase the SUV")



if __name__ == "__main__": #script executed from main_entry file
    app.run(debug=True) #Create a flask Local Server