from flask import Flask, render_template, request
import pandas as pd
from ml_model import run_ml_model
import pymongo
from pymongo import MongoClient

app = Flask(__name__)
MONGODB_HOST = "localhost"
MONGODB_PORT = 27017
MONGODB_DATABASE = "FashionStore"
MONGODB_COLLECTION = "Data"

client = MongoClient(MONGODB_HOST, MONGODB_PORT)
db = client[MONGODB_DATABASE]
collection = db[MONGODB_COLLECTION]

dbdata = []  

@app.route('/')
def home():
    return render_template('landing.html')

@app.route('/products', methods=['POST', 'GET'])
def products():
    global dbdata
    dbdata = list(collection.find({}))
    return render_template('products.html', data=dbdata)

@app.route('/sort', methods=['POST', 'GET'])
def sort():
    global dbdata
    run_ml_model()
    sort_order = request.form.get('sort-by')
    if sort_order == "trending":
        dbdata = sorted(dbdata, key=lambda x: x["rank"])
    elif sort_order == "lagging":
        dbdata = sorted(dbdata, key=lambda x: x["rank"], reverse=True)
    return render_template('products.html', data=dbdata)
@app.route('/about')
def about():
    return render_template('about.html')
@app.route('/contact')
def contact():
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)
