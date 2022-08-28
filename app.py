from flask import Flask,render_template,request
import pickle
import numpy as np
app=Flask(__name__)
with open("model.pickle","rb") as f:
    model=pickle.load(f)
with open("std.pickle","rb") as f:
    std=pickle.load(f)

@app.route("/",methods=['GET'])
def home():
    return render_template("index.html")
@app.route("/rent",methods=["POST"])
def pred():
    st1=float(request.form['st'])
    bed1=float(request.form['bed'])
    lt1=float(request.form['lt'])
    pt1=float(request.form['pt'])
    loc1=float(request.form['loc'])
    are=float(request.form['area'])
    ft1=float(request.form['ft'])

    data=np.array([[st1,bed1,lt1,pt1,loc1,are,ft1]])
    trans_data=std.transform(data)
    result=model.predict(trans_data)
    return render_template('index.html',prediction=result)

if __name__=='__main__':
    app.run(debug=True)