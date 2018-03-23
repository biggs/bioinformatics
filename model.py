from itertools import chain
import numpy as np
from Bio import SeqIO
from Bio.SeqUtils import GC, molecular_weight
from Bio.SeqUtils.ProtParam import ProteinAnalysis
from sklearn.model_selection import train_test_split
from sklearn.grid_search import GridSearchCV
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
from xgboost import plot_importance
from xgboost import DMatrix
from xgboost import train as xgtrain
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import RandomForestClassifier

np.random.seed(0)  #same splits and randomisation





class SeqsData(object):
    def __init__(self, names, directory='data', do_shuffle=True):
        self.names = names
        self.do_shuffle = do_shuffle
        self.files = ['{}/{}'.format(directory, name) for name in names]
        print("Extracting data from: {}".format(self.files))
        self.read_extract()
        self.feature_names = [
            'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N',
            'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Gravy',
            'Sequence Length', 'Sheet Fraction', 'Array Fraction',
            'Helix Fraction', 'Isoelectric Point', 'Instability Index',
            'Aromaticity', 'Molecular Weight']
        print("Data extracted and features assigned:\n{}".format(
            self.feature_names))


    def read_extract(self):
        """ Extract the files, features and split test and train. """
        fasta = [SeqIO.parse("{0}.fasta".format(f), "fasta")
                 for f in self.files]
        seqs = [[str(seq.seq) for seq in f]
                for f in fasta]

        n_label = [len(l) for l in seqs]
        y = np.array([i for i, n in enumerate(n_label) for _ in range(n)])
        X = self.seqs_to_features(seqs, sum(n_label))

        if self.do_shuffle:
            shuffle = np.random.permutation(X.shape[0])
            X, y = X[shuffle], y[shuffle]

        self.X = X
        self.y = y


    def seqs_to_features(self, seqs, no_seqs):
        """ Extract the features from the sequences."""
        X = np.zeros((no_seqs, 32))
        for i, s in enumerate(chain(*seqs)):  # iterate over all sequences
            # get amino acid counts
            alphabet = 'ABCDEFGHIKLMNPQRSTUVWXY' # no JOZ
            for j, letter in enumerate(alphabet):
                X[i, j] = s.count(letter)/len(s)

            # other analysis
            analysis = ProteinAnalysis(
                s.replace('X', 'A').replace('B', 'A').replace('U', 'A'))
            X[i, -1] = analysis.molecular_weight()
            X[i, -2] = analysis.aromaticity()
            X[i, -3] = analysis.instability_index()
            X[i, -4] = analysis.isoelectric_point()
            helix_array_sheet_fracs = analysis.secondary_structure_fraction()
            X[i, -5] = helix_array_sheet_fracs[0]
            X[i, -6] = helix_array_sheet_fracs[1]
            X[i, -7] = helix_array_sheet_fracs[2]
            X[i, -8] = len(s)
            X[i, -9] = analysis.gravy()  # mean hydrophobicity
        return X







def param_search(X_train, y_train):
    print("\nSearching Hyper-Parameter Space")
    cv_params = {'max_depth': [3, 4, 5, 6, 7], 'min_child_weight': [2, 3, 4, 5, 6]}
    optimized_GBM = GridSearchCV(XGBClassifier(**ind_params),
                            cv_params, scoring = 'accuracy',
                                 cv = 5, n_jobs = -1, verbose = 10)
    optimized_GBM.fit(X_train, y_train)
    print(optimized_GBM.grid_scores_)
    print(optimized_GBM.best_estimator_)



def predict_test(model, X_train, X_test, y_train, y_test):
    print("\nPredicting Test Labels")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print(np.mean(np.equal(preds, y_test)))


def predict_unseen(model, X, y, unseen):
    print("\nPredicting Unseen Labels")
    model.fit(X, y)
    preds = model.predict_proba(unseen)
    np.set_printoptions(threshold=np.nan, precision=3)
    print("Maximum Probability Class Label     Probability")
    for pred in preds:
        print("{}  {}".format(np.argmax(pred), np.max(pred)))


def cross_validate_metrics(model, X, y):
    """ Print a report of various metrics on cross validated
    predictions from the model.
    """
    print("\nCalculating Cross-Validated Metrics of Performance")
    y_pred = cross_val_predict(model, X, y, cv=5, verbose=2)
    target_names = ['cyto', 'mito', 'nucleus', 'secreted']
    np.set_printoptions(precision=3)
    print("Accuracy whole dataset after Cross Validation: {0}".format(
        accuracy_score(y, y_pred)))
    print(classification_report(y, y_pred, target_names=target_names))
    cm = confusion_matrix(y, y_pred)
    print(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis])
    print(cm)


def feature_importances_whole_dataset(X, y, feature_names):
    dtrain = DMatrix(X, label=y, feature_names=feature_names)
    params = {'max_depth': 7, 'min_child_weight': 2, 'num_class': 4, **ind_params}
    model = xgtrain(params, dtrain)
    fig, ax = plt.subplots(figsize=(10,8))
    plot_importance(model, height=0.8, ax=ax, grid=False, importance_type='gain')


def random_forest_baseline(X, y):
    model = RandomForestClassifier(n_estimators=100)
    cross_validate_metrics(model, X, y)



def cross_validate_roc(model, X, y):
    y_pred = cross_val_predict(model, X, y, cv=5, verbose=2, method='predict_proba')
    target_names = ['Cytosolic', 'Mitochondrial', 'Nuclear', 'Secreted']
    for i, n in enumerate(target_names):
        print(i, n)
        fpr, tpr, threshold = roc_curve(y==i, y_pred[:,i])
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.title('Receiver Operating Characteristic, {}'.format(n))
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')





if __name__=='__main__':
    # TODO: number estimators back to 1000
    ind_params = {'learning_rate': 0.01, 'seed':0, 'subsample': 0.8, 'n_estimators': 100,
                'colsample_bytree': 0.8, 'objective': 'multi:softmax'}


    # Extract files and features
    names = ['cyto', 'mito', 'nucleus', 'secreted']
    data = SeqsData(names)
    unseen_data = SeqsData(['blind'], do_shuffle=False)

    # Original test train splits for holding out before hyper-parameter optimisation
    X_train, X_test, y_train, y_test = train_test_split(data.X, data.y,
                                                        test_size=0.2, random_state=0)
    train = {'X': X_train, 'y': y_train}
    test = {'X': X_test, 'y': y_test}


    # EVERYTHING USED IN THE PAPER:
    # param_search(train['X'], train['y'])  # parameter search not using test set
    xgb = XGBClassifier(max_depth = 7, min_child_weight = 2, **ind_params)
    predict_test(xgb, train['X'], test['X'], train['y'], test['y'])
    # predict_unseen(xgb, data.X, data.y, unseen_data.X)
    # cross_validate_metrics(xgb, data.X, data.y)
    # cross_validate_roc(xgb, data.X, data.y)
    # feature_importances_whole_dataset(data.X, data.y, data.feature_names)

    # random_forest_baseline(data.X, data.y)
    plt.show()
