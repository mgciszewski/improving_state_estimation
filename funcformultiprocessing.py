from postprocessingbyprojection import *
import numpy as np
from timeit import default_timer as timer
from BSS import *

def timing_mean_DAG(gam, size):
    jumps = np.random.uniform(0, gam, size).cumsum().tolist()
    start1 = timer()
    g = DAG(jumps, gam)
    end1 = timer()
    start2 = timer()
    g.shortestPath()
    end2 = timer()
    return (size, end1-start1, end2-start2)

def size_DAG(gam, size):
    jumps = np.random.uniform(0, gam, size).cumsum().tolist()
    g = DAG(jumps, gam)
    return (size, sum(sum(g.weightMatrix != np.inf)) + g.number_of_vertices)

def crossvalsubset(subset, train_df, train_fit_array, train_test_array, transform_to_std, clf, meas):
    train_lts = []
    train_ltsplus = []
    test_lts = []
    test_ltsplus = []
    for i in range(10):
        train_fit = train_fit_array[i]
        train_test = train_test_array[i]
        indices_fit = train_df['trial'].isin(train_fit)
        X = train_df.loc[indices_fit, ['trial'] + list(subset)].reset_index(drop=True)
        X_std = transform_to_std.fit_transform(X)
        y = train_df.loc[indices_fit, 'manual_label'].reset_index(drop=True)
        y = (y != 0).astype(int).reset_index(drop=True)

        clf.fit(X_std,y)
        current_lts = []
        current_ltsplus = []

        for j in range(10):
            indices_train = (train_df['trial'] == train_fit[j])
            X_test = train_df.loc[indices_train, ['trial'] + list(subset)].reset_index(drop=True)
            X_test_std = transform_to_std.transform(X_test)
            y_test = train_df.loc[indices_train, 'manual_label'].reset_index(drop=True)
            y_test = (y_test != 0).astype(int).reset_index(drop=True)
            prediction = clf.predict(X_test_std)

            BSS = BinaryStateSequence(series = prediction)
            ppBSS = BSS.find_approx(0.5)
            groundtruth = BinaryStateSequence(series = y_test)

            current_lts.append(meas.performance(groundtruth, BSS))
            current_ltsplus.append(meas.performance(groundtruth, ppBSS))

        train_lts.append(current_lts)
        train_ltsplus.append(current_ltsplus)

        current_lts = []
        current_ltsplus = []

        for j in range(5):
            indices_test = (train_df['trial'] == train_test[j])
            X_test = train_df.loc[indices_test, ['trial'] + list(subset)].reset_index(drop=True)
            X_test_std = transform_to_std.transform(X_test)
            y_test = train_df.loc[indices_test, 'manual_label'].reset_index(drop=True)
            y_test = (y_test != 0).astype(int).reset_index(drop=True)
            prediction = clf.predict(X_test_std)

            BSS = BinaryStateSequence(series = prediction)
            ppBSS = BSS.find_approx(0.5)
            groundtruth = BinaryStateSequence(series = y_test)

            current_lts.append(meas.performance(groundtruth, BSS))
            current_ltsplus.append(meas.performance(groundtruth, ppBSS))

        test_lts.append(current_lts)
        test_ltsplus.append(current_ltsplus)
    return subset, train_lts, train_ltsplus, test_lts, test_ltsplus