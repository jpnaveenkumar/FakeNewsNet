from pyspark.sql import SparkSession
import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from xgboost import XGBClassifier
models_obj = None
class models:
    KNN = None
    LR = None
    SGD = None
    DecisionTree = None
    RandomForest = None
    NaiveBayes = None
    XGB = None
    modelAccuray = dict()

    def init_knn(self):
        self.KNN = KNeighborsClassifier(n_neighbors=1)
        return self.get_Knn()
    
    def init_LR(self):
        self.LR = LogisticRegression()
        return self.get_LR()
    
    def init_SGD(self):
        self.SGD = SGDClassifier()
        return self.get_SGD()
    
    def init_DecisionTree(self):
        self.DecisionTree = DecisionTreeClassifier()
        return self.get_DecisionTree()

    def init_RandomForest(self):
        self.RandomForest = RandomForestClassifier(n_estimators=100,
                                bootstrap = True,
                                max_features = 'sqrt')
        return self.get_RandomForestClassifier()

    def init_GaussianNB(self):
        self.NaiveBayes = GaussianNB()
        return self.get_NaiveBayes()

    def init_XGB(self):
        pars = {'random_state' : 3, 'n_jobs' : 4}
        xgb = XGBClassifier(**pars)
        # xgb_params = {
        #     'learning_rate'		: [0.1, 0.05, 0.01],
        #     'max_depth'			: [3, 5, 7],
        #     'n_estimators'		: [50, 100, 200, 300, 400],
        # }
        # xgbCV = GridSearchCV(xgb, xgb_params, cv=5, verbose=2, scoring='roc_auc', return_train_score=True)
        # self.XGB = xgbCV
        self.XGB = xgb
        return self.get_XGBoost()

    def set_model_accuracy(self,dic):
        self.modelAccuray = dic

    def get_model_accuracy(self):
        return self.modelAccuray
    
    def get_Knn(self):
        return self.KNN
    
    def get_LR(self):
        return self.LR
    
    def get_SGD(self):
        return self.SGD
    
    def get_DecisionTree(self):
        return self.DecisionTree

    def get_RandomForestClassifier(self):
        return self.RandomForest

    def get_NaiveBayes(self):
        return self.NaiveBayes

    def get_XGBoost(self):
        return self.XGB
    
def get_models_obj():
    global models_obj 
    if(models_obj == None):
        models_obj = models()
    return models_obj
def stringToArray(array):
    print(array)
    return [int(i) for i in array[1:-1].split(",") if i != '']
spark = SparkSession \
	.builder \
	.appName("Bigdata Project") \
	.getOrCreate()
def get_spark_session():
    return spark
def init_all_models():
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

    model_obj = get_models_obj()
    accuracy = dict()
    
    nb = model_obj.init_GaussianNB()
    nb.fit(x_train, y_train)
    y_pred = nb.predict(x_test)
    accuracy["naiveBayes"] = accuracy_score(y_test, y_pred)
    print('GaussianNB Accuracy :',accuracy_score(y_test, y_pred))

    tree = model_obj.init_DecisionTree()
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    accuracy["decisionTree"] = accuracy_score(y_test, y_pred)
    print('DecisionTreeClassifier Accuracy :',accuracy_score(y_test, y_pred))

    model = model_obj.init_RandomForest()
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    accuracy["randomForest"] = accuracy_score(y_test, y_pred)
    print("RandomForestClassifier Accuracy :", accuracy_score(y_test, y_pred))

    LR = model_obj.init_LR()
    LR.fit(x_train, y_train)
    y2_LR_model = LR.predict(x_test)
    accuracy["logisticRegression"] = accuracy_score(y_test, y2_LR_model)
    print("LR Accuracy :", accuracy_score(y_test, y2_LR_model))

    SDG = model_obj.init_SGD()
    SDG.fit(x_train, y_train)
    y2_SDG_model = SDG.predict(x_test)
    print("SDG Accuracy :", accuracy_score(y_test, y2_SDG_model))

    KNN = model_obj.init_knn()
    KNN.fit(x_train,y_train)
    y2_KNN_model = KNN.predict(x_test)
    accuracy["knn"] = accuracy_score(y_test, y2_KNN_model)
    print("KNN Accuracy :", accuracy_score(y_test, y2_KNN_model))

    XGB = model_obj.init_XGB()
    XGB.fit(np.array(x_train),y_train)
    y2_XGB_model = XGB.predict(np.array(x_test))
    accuracy["XGB"] = accuracy_score(y_test, y2_XGB_model)
    print("XGB Accuracy :", accuracy_score(y_test, y2_XGB_model))

    model_obj.set_model_accuracy(accuracy)