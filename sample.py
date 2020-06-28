from flask import Flask
from pyspark.sql import SparkSession
import pandas as pd
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
def stringToArray(array):
    print(array)
    return [int(i) for i in array[1:-1].split(",") if i != '']
def getAccuracy():
    features_pd = pd.read_csv('features.csv')
    feature_ppd = spark.createDataFrame(data=features_pd)
    features_rdd = feature_ppd.rdd
    features_rdd_v1 = features_rdd.map(lambda x: (x[1],stringToArray(x[2]),[x[3]],stringToArray(x[4])))
    features_rdd_v2 = features_rdd_v1.map(lambda x: (x[0],x[1]+x[3]+x[2]))

    consolidated_pd = pd.read_csv('consolidated_dataset.csv')
    consolidated_pd_v1 = consolidated_pd[['news_url','credibility']]
    consolidated_pd_v1[['news_url','credibility']] = consolidated_pd_v1[['news_url','credibility']].astype(str)
    consolidated_ppd = spark.createDataFrame(data=consolidated_pd_v1)
    consolidated_rdd = consolidated_ppd.rdd

    combined = consolidated_rdd.join(features_rdd_v2)
    features_x = combined.map(lambda row: (row[1][1],))
    features_y = combined.map(lambda row: (row[1][0],))
    features_x_df = features_x.toDF()
    features_y_df = features_y.toDF()
    features_x_list = []
    features_y_list = []
    for row in features_x_df.take(10460):
        features_x_list.append(list(row)[0])
    for row in features_y_df.take(10460):
        features_y_list.append(list(row)[0])
    x_train, x_test, y_train, y_test = train_test_split(features_x_list, features_y_list, test_size=0.2, random_state=3, stratify=features_y_list)
    nb = GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    return str(accuracy)
app = Flask(__name__)
spark = SparkSession \
	.builder \
	.appName("Bigdata Assingnment - 2") \
	.config("writer", "NAVEEN KUMAR JAKUVA PREMKUMAR") \
	.getOrCreate()
def process_data():
    data = spark.sparkContext.textFile("consolidated_dataset.csv",2000)
    print(data.take(4))
@app.route("/")
def hello():
    #word_document_vs_count = spark.sparkContext.parallelize(["hii","this"])
    #print(word_document_vs_count.collect())
    #process_data();
    return getAccuracy()
if __name__ == "__main__":
    app.run(debug=True)
