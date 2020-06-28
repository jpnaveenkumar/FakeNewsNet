from flask import Flask, render_template, request
from pyspark.sql import SparkSession
import pandas as pd
from service import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
app = Flask(__name__)

@app.route("/FakeNewsNet")
def index():
    return render_template("index.html")

@app.route("/")
def hello():
    return get_credibility()

@app.route("/getCredibility")
def getCredibility():
    url = request.args.get("url")
    print(url)
    return get_credibility(url)
    #result = get_credibility(url)
    #resp = dict()
    #resp["credibility"] = result
    #return result
if __name__ == "__main__":
    global cache
    init_models()
    app.run(debug=True)
