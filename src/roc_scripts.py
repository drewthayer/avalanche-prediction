import pickle
import numpy as np
import matplotlib.pyplot as plt

def calc_roc(prediction_output, thresholds):
    #thresholds = np.linspace(0.1,6,100)
    TP_rates = []
    FP_rates = []
    accuracies = []
    precisions = []
    recalls = []
    threshold_list = []
    cf_list = []
    for threshold in thresholds:
        output = np.where(prediction_output >= threshold, 1, 0)
        true = output[:,0]
        predicted = output[:,1]
        TP = np.sum(np.logical_and(predicted==1, true==1))
        TN = np.sum(np.logical_and(predicted==0, true==0))
        FP = np.sum(np.logical_and(predicted==1, true==0))
        FN = np.sum(np.logical_and(predicted==0, true==1))
        TP_rates.append(TP/len(true))
        FP_rates.append(FP/len(true))

        accuracy = (TP+TN)/(TP+TN+FP+FN)
        precision = TP/(TP+FP)
        recall = TP/(TP+FN)
        threshold_list.append(threshold)
        accuracies.append(accuracy)
        recalls.append(recall)
        precisions.append(precision)

        #confustion matrix
        cf_list.append([TP,TN,FP,FN])

    return TP_rates, FP_rates, accuracies, precisions, recalls, threshold_list, cf_list

if __name__=='__main__':
    # load data
    output_rfr = pickle.load(open('output_rfr.p', 'rb'))
    output_gbr = pickle.load(open('output_rgbr.p', 'rb'))

    # run ROC calculator
    thresholds = np.linspace(0.01,1,100)
    TPr_rfr, FPr_rfr, acc_rfr, prec_rfr, rec_rfr, thresh_rfr, cf_list = calc_roc(output_rfr, thresholds)
    TPr_gbr, FPr_gbr, acc_gbr, prec_gbr, rec_gbr, thresh_gbr, cf_list = calc_roc(output_gbr, thresholds)

    # roc plot
    plt.plot(FPr_rfr, TPr_rfr, 'b', label='rfr')
    plt.plot(FPr_gbr, TPr_gbr, 'teal', label='gbr')
    plt.xlabel('False Positive rate')
    plt.ylabel('True Positive rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend()
    plt.savefig('ROC_rfr_gbr_t1.png',dpi=250)
    plt.close()

    # accuracy, recall, precision plot
    plt.plot(thresh_rfr, acc_rfr, 'b', label='accuracy')
    plt.plot(thresh_rfr, rec_rfr, 'g', label='recall')
    plt.plot(thresh_rfr, prec_rfr, 'magenta', label='precision')
    plt.xlabel('probability threshold')
    plt.legend()
    plt.title('random forest model')
    plt.savefig('acc_rec_prec_rfr_t1.png',dpi=250)
    plt.close()

    # confusion matrix at a specific threshold:
    thresholds = [0.46]
    TPr_rfr, FPr_rfr, acc_rfr, prec_rfr, rec_rfr, thresh_rfr, cf_list = calc_roc(output_rfr, thresholds)
    print('threshold = {}'.format(thresh_rfr[0]))
    print('accuracy = {:0.3f}'.format(acc_rfr[0]))
    print('precision = {:0.3f}'.format(prec_rfr[0]))
    print('recall= {:0.3f}'.format(rec_rfr[0]))

    print(cf_list[0])
