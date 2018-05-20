import numpy as np
import matplotlib.pyplot as plt

class ROC():

    '''
    y_true: np array, 0 or 1
    y_hat: np array, 0 or 1
    '''

    def __init__(self):
        self.blah = 'blah'
        self.TP = None
        self.TN = None
        self.FP = None
        self.FN = None
        self.TPR = None
        self.FPR = None
        self.accuracy = None
        self.precision = None
        self.recall = None

    def fit(self, y_true, y_hat):
        self.TP = np.sum(np.add(y_true, y_hat)==2)
        self.TN = np.sum(np.add(y_true, y_hat)==0)
        self.FP = np.sum(np.subtract(y_true, y_hat)==-1)
        self.FN = np.sum(np.subtract(y_true, y_hat)==1)
        self.TPR = self.TP / (self.TP + self.FN)
        self.FPR = self.FN / (self.TP + self.FN)
        self.accuracy = (self.TP + self.TN) / (self.TP + self.TN + self.FP + self.FN)
        self.recall = self.TPR
        self.precision = self.TP/(self.TP + self.FP)

    def confusion_mtx(self):
        return (self.TP, self.TN, self.FP, self.FN)

    def standard_confusion_matrix(self, y_true, y_predict):
        """
        creates standard confusion matrix in the form of [[tp, fp], [fn, tn]]
        inputs: 2x np.arrays
        output: 1x np.array
        """
        tp = np.sum(np.add(y_true, y_predict)==2)
        tn = np.sum(np.add(y_true, y_predict)==0)
        fp = np.sum(np.subtract(y_true, y_predict)==-1)
        fn = np.sum(np.subtract(y_true, y_predict)==1)
        return tp, fp, fn, tn

    def roc_plot(self, y_true, y_proba, color, label, filename, dpi=250):
        fig, ax = plt.subplots()
        FP_rates = []
        TP_rates = []
        #make thresholds
        thresholds = y_proba
        thresholds.sort()
        thresholds = np.flip(thresholds, axis=0)
        thresholds = np.insert(thresholds,0,1)
        for t in thresholds:
            y_temp = np.where(y_proba > t, 1, 0)
            tp, fp, fn, tn = self.standard_confusion_matrix(y_true, y_temp)
            FP_rates.append(fp/(fp+tn))
            TP_rates.append(tp/(tp+fn))
        plt.plot(FP_rates, TP_rates, color, label=label)
            #plt.plot(FPRs_s, TPRs_s, 'teal', label='SVD++')
        plt.xlabel('False Positive rate')
        plt.ylabel('True Positive rate')
        plt.title('Receiver Operating Characteristic')
        plt.legend()
        #plt.savefig(filename,dpi=dpi)
        #plt.close()
        plt.show()
        return FP_rates, TP_rates, t

    def a_r_p_plot(self, y_true, y_proba):
        fig, ax = plt.subplots()
        accs = []
        prec = []
        recs = []

        thresholds = y_proba
        thresholds.sort()
        thresholds = np.flip(thresholds, axis=0)
        thresholds = np.insert(thresholds,0,1)
        for t in thresholds:
            y_temp = np.where(y_proba > t, 1, 0)
            tp, fp, fn, tn = self.standard_confusion_matrix(y_true, y_temp)
            accs.append((tp+fp)/(tp+tn+fp+fn))
            prec.append(tp/(tp+fp))
            recs.append(tp/(tp+fn))
        plt.plot(thresholds, accs, 'b', label='acc')
        plt.plot(thresholds, prec, 'k--', label='prec')
        plt.plot(thresholds, recs, 'g--', label='recall')
        plt.xlabel('probability threshold')
        plt.legend()
        plt.show()










#     def calc_for_roc(dataframe, thresholds):
#     TPs = []
#     TNs = []
#     FPs = []
#     FNs = []
#     recalls = []
#     precisions = []
#     accuracies = []
#     for t in thresholds:
#     df = calc_confusion(dataframe, t)
#     c_mtx = confusion_mtx(df)
#     TPs.append(c_mtx[0])
#     TNs.append(c_mtx[1])
#     FPs.append(c_mtx[2])
#     FNs.append(c_mtx[3])
#     recalls.append(c_mtx[3]/(c_mtx[1]+c_mtx[3]))
#     precisions.append(c_mtx[2]/(c_mtx[1]+c_mtx[2]))
#     accuracies.append((c_mtx[0]+c_mtx[1])/(c_mtx[0]+c_mtx[1]+c_mtx[2]+c_mtx[3]))
#
#     FPRs = np.array(FPs)/(np.array(FPs)+np.array(TNs))
#     TPRs = np.array(TPs)/(np.array(TPs)+np.array(FNs))
#
#     return TPs, TNs, FPs, FNs, FPRs, TPRs, recalls, precisions, accuracies
