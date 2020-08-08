import pickle
from flask import Flask,render_template,url_for,request

# load the model from disk
clf = pickle.load(open('model.pkl', 'rb'))
cv=pickle.load(open('fittranform.pkl','rb'))
app = Flask(__name__)

@app.route('/')
def home():
	return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():       
    if request.method == 'POST':
        query = request.form['query']
        data = [query]
        vect = cv.transform(data).toarray()
        my_prediction = clf.predict(vect)
    return render_template('result.html',prediction = my_prediction)

if __name__ == '__main__':
	app.run(debug=True)