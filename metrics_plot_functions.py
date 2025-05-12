from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, matthews_corrcoef, f1_score, classification_report, accuracy_score, recall_score, precision_score, roc_auc_score
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Oranges):
    """
    DEPRECATED!
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.figure(figsize = (10, 10))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title, size = 24)
    plt.colorbar(aspect=4)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45, size = 14)
    plt.yticks(tick_marks, classes, size = 14)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    
    # Labeling the plot
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), fontsize = 20,
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
        
    plt.grid(None)
    plt.tight_layout()
    plt.ylabel('True label', size = 18)
    plt.xlabel('Predicted label', size = 18)
    

def output_confusion_matrix(train_or_test:str, actual_vals:pd.DataFrame, predicted_vals:pd.DataFrame, display_labels:list, title:str):
    """Output a confusion matrix

    Args:
        model_type (str): Train or Test
        actual_vals (pd.DataFrame): e.g. y_test
        predicted_vals (pd.DataFrame): e.g. y_test_pred
        display_labels (list): e.g. ['Not Fraud', 'Fraud'] corresponding to 0, 1 in y vals.
        title (str): Title for chart
    """
    cm = ConfusionMatrixDisplay(confusion_matrix(actual_vals, predicted_vals, normalize='all'),
                                display_labels=display_labels
                                )

    cm.plot()

    plt.title(f'Confusion Matrix ({train_or_test})\n{title}')
    plt.show()
    
def print_metric_stats(title:str, train_or_test:str, actual_vals:pd.DataFrame, predicted_vals:pd.DataFrame):
    """Print summary statistics for model.

    Args:
        title (str): Info on model
        train_or_test (str): Train or Test
        actual_vals (pd.DataFrame): e.g. y_test
        predicted_vals (pd.DataFrame): e.g. y_test_pred
    """

    print(f'```\nModel performance for\n {title}:\n---------------')
    print(f'* {train_or_test} F1 Score:  {round(f1_score(actual_vals, predicted_vals), 4)}')
    print(f'* {train_or_test} ROC AUC:   {round(roc_auc_score(actual_vals, predicted_vals), 4)}')
    print(f'* {train_or_test} MCC:       {round(matthews_corrcoef(actual_vals, predicted_vals), 4)}')
    print(f'* {train_or_test} Accuracy:  {round(accuracy_score(actual_vals, predicted_vals), 4)}')
    print(f'* {train_or_test} Precision: {round(precision_score(actual_vals, predicted_vals), 4)}')
    print(f'```')

