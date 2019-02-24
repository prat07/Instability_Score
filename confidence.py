import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
from sklearn.datasets import make_classification
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy.linalg

# eps parameter is tunable
def geometric_median(X, eps=1e-10):
    y = np.mean(X, 0)

    while True:
        D = cdist(X, [y])
        nonzeros = (D != 0)[:, 0]

        Dinv = 1 / D[nonzeros]
        Dinvs = np.sum(Dinv)
        W = Dinv / Dinvs
        T = np.sum(W * X[nonzeros], 0)

        num_zeros = len(X) - np.sum(nonzeros)
        if num_zeros == 0:
            y1 = T
        elif num_zeros == len(X):
            return y
        else:
            R = (T - y) * Dinvs
            r = np.linalg.norm(R)
            rinv = 0 if r == 0 else num_zeros/r
            y1 = max(0, 1-rinv)*T + min(1, rinv)*y

        if euclidean(y, y1) < eps:
            return y1

        y = y1

def get_accuracy_score_test_set(model, X_test, y_test, threshold):
    print("Accuracy at threshold {0} is {1}".format(threshold, accuracy_score(y_test,(model.predict_proba(X_test)[:,1] >= threshold).astype(int))))
    return accuracy_score(y_test,(model.predict_proba(X_test)[:,1] >= threshold).astype(int))

def get_geometric_median_TP_TN_FP_FN(model, X_train, y_train, threshold):
    index_TP = []
    index_TN = []
    index_FP = []
    index_FN = []
    gm = {'TP':0, 'TN':0, 'FP':0, 'FN':0}

    for index in range(len(X_train)):
        p = (model.predict_proba(X_train)[:,1] >= threshold).astype(int)[index]
        a = y_train[index]
        if (p == a):
            if p == 1:
                index_TP.append(index)
            if p == 0:
                index_TN.append(index)
        if (p != a):
            if p == 1:
                index_FP.append(index)
            if p == 0:
                index_FN.append(index)

    gm['TP'] = geometric_median(X_train[np.array(index_TP)])
    gm['TN'] = geometric_median(X_train[np.array(index_TN)])
    gm['FP'] = geometric_median(X_train[np.array(index_FP)])
    gm['FN'] = geometric_median(X_train[np.array(index_FN)])

    return gm

def create_df_with_confidence_and_instability_score(model, X_test, y_test, threshold, gm):
    dist_correct = []
    dist_incorrect = []
    actual = []
    predicted = []
    confidence = []
    instability_score = []

    for index in range(len(X_test)):

        p = (model.predict_proba(X_test)[:,1] >= threshold).astype(int)[index]

        if p == 1:
            dc = np.linalg.norm(gm['TP'] - X_test[index])
            dist_correct.append(dc)
            di = np.linalg.norm(gm['FP'] - X_test[index])
            dist_incorrect.append(di)
            if dc <= di :
                confidence.append('High')
            if dc > di :
                confidence.append('Low')
            instability_score.append(float(dc/di))
            predicted.append(1)
            actual.append(y_test[index])


        if p == 0:
            dc = np.linalg.norm(gm['TN'] - X_test[index])
            dist_correct.append(dc)
            di = np.linalg.norm(gm['FN'] - X_test[index])
            dist_incorrect.append(di)
            if dc <= di :
                confidence.append('High')
            if dc > di :
                confidence.append('Low')
            instability_score.append(float(dc/di))
            predicted.append(0)
            actual.append(y_test[index])

    df = pd.DataFrame({'Predicted': predicted, 'Dist_Correct': dist_correct, 'Dist_Incorrect': dist_incorrect, 'Actual' : actual, 'Confidence' : confidence, 'Instability_Score': instability_score})
    return df

def get_accuracy_with_high_confidence(df_high):
    mean = df_high['Instability_Score'].mean()
    std = df_high['Instability_Score'].std()
    # Tunable parameter
    if len(df_high[(df_high.Instability_Score < (mean - 3 * std ))]) != 0:
        df_high = df_high[(df_high.Instability_Score < (mean - 3 * std ))]
    elif len(df_high[(df_high.Instability_Score < (mean - 2 * std ))]) != 0:
        df_high = df_high[(df_high.Instability_Score < (mean - 2 * std ))] 
    elif len(df_high[(df_high.Instability_Score < (mean - 1 * std ))]) != 0:
        df_high = df_high[(df_high.Instability_Score < (mean - 1 * std ))] 
    else:
        df_high = df_high
    print "Size of df_high", len(df_high)
    return accuracy_score(df_high['Actual'], df_high['Predicted'])



def get_accuracy_with_low_confidence(df_low):
    mean = df_low['Instability_Score'].mean()
    std = df_low['Instability_Score'].std()
     # Tunable parameter
    if len(df_low[(df_low.Instability_Score > (mean + 3 * std))]) != 0:
        df_low = df_low[(df_low.Instability_Score > (mean + 3 * std))]
    elif len(df_low[(df_low.Instability_Score > (mean + 2 * std))]) != 0:
        df_low = df_low[(df_low.Instability_Score > (mean + 2 * std))]
    elif len(df_low[(df_low.Instability_Score > (mean + 1 * std))]) != 0:
        df_low = df_low[(df_low.Instability_Score > (mean + 1 * std))]
    else:
        df_low = df_low
    print "Size of df_low", len(df_low)
    return accuracy_score(df_low['Actual'], df_low['Predicted'])

def is_valid_threshold(model, X_train, y_train, threshold):
    index_TP = []
    index_TN = []
    index_FP = []
    index_FN = []

    for index in range(len(X_train)):
        p = (model.predict_proba(X_train)[:,1] >= threshold).astype(int)[index]
        a = y_train[index]
        if (p == a):
            if p == 1:
                index_TP.append(index)
            if p == 0:
                index_TN.append(index)
        if (p != a):
            if p == 1:
                index_FP.append(index)
            if p == 0:
                index_FN.append(index)

    if len(index_TP) == 0:
        return 0
    if len(index_TN) == 0:
        return 0
    if len(index_FP) == 0:
        return 0
    if len(index_FN) == 0:
        return 0
    else:
        return 1

def get_all_accuracy_set_with_thresholds(model, X_train, y_train, X_test, y_test):
    valid_threshold = []
    full_threshold = []
    accuracy_model = []

    # Find valid thresholds i.e. when TP, TN, FP, FN are defined
    for threshold in np.arange(0.0, 1.00, 0.01):
        full_threshold.append(threshold)
        accuracy_model.append(get_accuracy_score_test_set(model, X_test, y_test, threshold))
        if (is_valid_threshold(model, X_train, y_train, threshold)):
                valid_threshold.append(threshold)
    print valid_threshold
    print("valid threshold range {0} to {1}".format(valid_threshold[0], valid_threshold[-1])) 

    accuracy_low_conf = []
    accuracy_high_conf = []
    conf_threshold = []

    # Find all accuarcy set for all valid thresholds with step = 0.01
    step = 0.01
    for threshold in np.arange(valid_threshold[0], valid_threshold[-1], step):
        print "Threshold = ", threshold 
        conf_threshold.append(threshold)
        gm = get_geometric_median_TP_TN_FP_FN(model, X_train, y_train, threshold)
        df = create_df_with_confidence_and_instability_score(model, X_test, y_test, threshold, gm)
        df_high = df[(df.Confidence == 'High')]
        accuracy_high_conf.append(get_accuracy_with_high_confidence(df_high))
        df_low = df[(df.Confidence == 'Low')]
        accuracy_low_conf.append(get_accuracy_with_low_confidence(df_low))
        
    return full_threshold, conf_threshold, accuracy_model, accuracy_high_conf, accuracy_low_conf

def perform_logistic_regression(X_train, y_train):
    from sklearn.linear_model import LogisticRegression
    model = LogisticRegression(C=0.5) # needed bcz of over-fitting issues
    model.fit(X_train, y_train)
    return model

def perform_gaussian_nb(X_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    model = GaussianNB()
    model.fit(X_train, y_train)
    return model

def perform_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(max_depth=3) # needed bcz of over-fitting issues
    model.fit(X_train, y_train)
    return model

def perform_neural_network(X_train, y_train):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier()
    model.fit(X_train, y_train)
    return model

def create_subplot(full_threshold, conf_threshold, accuracy_model, accuracy_high_conf, accuracy_low_conf):

    plt.plot(conf_threshold,accuracy_high_conf, 'g--', label = "High Confidence Model Accuracy")
    plt.plot(full_threshold,accuracy_model, 'k--', label = "Baseline Model Accuracy")
    plt.plot(conf_threshold,accuracy_low_conf, 'b--', label = "Low Confidence Model Accuracy")

    plt.axvline(conf_threshold[0], color='r')
    plt.axvline(conf_threshold[-1], color='r')

    plt.xlabel('Thresholds')
    plt.ylabel('Accuracy')
