# Purpose - Receive the call for testing a page from the Chrome extension and return the result (SAFE/PHISHING)
# for display. This file calls all the different components of the project (The ML model, features_extraction) and
# consolidates the result.

import features_extraction
import sys
import numpy as np
import arraystore

#from features_extraction import LOCALHOST_PATH, DIRECTORY_NAME
# from sklearn.externals import joblib


def get_prediction_from_url(test_url):
    features_test = features_extraction.main(test_url)
    # Due to updates to scikit-learn, we now need a 2D array as a parameter to the predict function.
    features_test = np.array(features_test).reshape((1, -1))
    #print ("The probability of this site being a phishing website is ", features_test[0]*100, "%")

    clf = joblib.load('classifier/elm.pkl')
    #print(type(features_test))
    pred = clf.predict(features_test)
    newar=features_test[0]

    ij= int(arraystore.mainp(newar))
    if(ij==-1) :
        return int(-1)
    else:
        return int(1)
    #return int(pred[0])


def main():
    url = sys.argv[1]
    prediction = get_prediction_from_url(url)

    #prediction = get_prediction_from_url("https://www.kaggle.com/c/caltech-cs-155-2018")

    #Print the probability of prediction (if needed)
    #prob = clf.predict_proba(features_test)
    # print 'Features=', features_test, 'The predicted probability is - ', prob, 'The predicted label is - ', pred
    
    if prediction == 1:
        print("\n")
        print ("\n The website is safe to browse")
        print(" ")
        #print("SAFE")
    elif prediction == -1:
        print("")
        print ("\n The website has phishing features. \n  DO NOT VISIT!")
        #print("PHISHING")
        print("")

        # print 'Error -', features_test


if __name__ == "__main__":
    main()
