from dataProcessing_perNews import *
from model import *
import numpy as np

def get_test_features(newsLink):
    out = extractFeatures(newsLink)
    print(out[0][1:])
    result = out[0][1:]
    test_features = result[0] + result[2] + [result[1]]
    print(test_features)
    return test_features

def init_models():
    init_all_models()

def get_credibility(url):
    resp = dict()
    model_result = dict()
    result = []
    #"www.dailymail.co.uk/tvshowbiz/article-5874213/Did-Miley-Cyrus-Liam-Hemsworth-secretly-married.html" == > fake
    test_features = get_test_features(url)
    #test_features = get_test_features('https://www.straitstimes.com/singapore/coronavirus-record-120-new-covid-19-cases-in-spore-two-foreign-worker-dormitories-gazetted')
    models_obj = get_models_obj()
    accuracy = models_obj.get_model_accuracy() 
    knn = models_obj.get_Knn()
    lr = models_obj.get_LR()
    decisionTree = models_obj.get_DecisionTree()
    randomForest = models_obj.get_RandomForestClassifier()
    naiveBayes = models_obj.get_NaiveBayes()
    sgd = models_obj.get_SGD()
    xgb = models_obj.get_XGBoost()
    result.append(knn.predict([test_features])[0])
    model_result["knn"] = result[0]
    #print(result)
    result.append(lr.predict([test_features])[0])
    model_result["lr"] = result[1]
    #print(result)
    result.append(decisionTree.predict([test_features])[0])
    model_result["decisionTree"] = result[2]
    #print(result)
    result.append(randomForest.predict([test_features])[0])
    model_result["randomForest"] = result[3]
    #print(result)
    result.append(naiveBayes.predict([test_features])[0])
    model_result["naiveBayes"] = result[4]
    #print(result)
    result.append(sgd.predict([test_features])[0])
    #print(result)
    result.append(xgb.predict(np.array([test_features]))[0])
    model_result["xgb"] = result[6]
    resp["model_result"] = model_result;
    resp["accuracy"] = accuracy;
    print(result)
    if( result.count('fake') > result.count('real')):
        print('fake')
        resp["credibility"] = 'fake'
    else:
        print('real')
        resp["credibility"] = 'real'
    print(resp)
    return resp

#init_models()
#get_credibility()
#get_test_features("www.dailymail.co.uk/tvshowbiz/article-5874213/Did-Miley-Cyrus-Liam-Hemsworth-secretly-married.html")