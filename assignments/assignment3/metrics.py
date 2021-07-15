def binary_classification_metrics(prediction, ground_truth):
    '''
    Computes metrics for binary classification

    Arguments:
    prediction, np array of bool (num_samples) - model predictions
    ground_truth, np array of bool (num_samples) - true labels

    Returns:
    precision, recall, f1, accuracy - classification metrics
    '''
    precision = 0
    recall = 0
    accuracy = 0
    f1 = 0

    tp, tn, fp, fn = 0, 0, 0, 0
    
    for pr, gt in zip(prediction, ground_truth):
        if pr == gt:
            if gt == True:
                tp += 1
            else:
                tn += 1
        else:
            if gt == True:
                fn += 1
            else:
                fp += 1
          
    try:
        precision = tp / (tp + fp)
    except ZeroDivisionError:
        print('precision = inf')
        
    try:
        recall = tp / (tp + fn)
    except ZeroDivisionError:
        print('recall = inf')
        
    accuracy = (tp + tn) / (tp + tn + fn + fp)
    
    try:
        f1 = 2 * precision * recall / (precision + recall)
    except ZeroDivisionError:
        print('f1 = 0')
        f1 = 0
        
    # Some helpful links:
    # https://en.wikipedia.org/wiki/Precision_and_recall
    # https://en.wikipedia.org/wiki/F1_score
    
    return precision, recall, accuracy, f1


def multiclass_accuracy(prediction, ground_truth):
    '''
    Computes metrics for multiclass classification

    Arguments:
    prediction, np array of int (num_samples) - model predictions
    ground_truth, np array of int (num_samples) - true labels

    Returns:
    accuracy - ratio of accurate predictions to total samples
    '''
    
    accuracy = 0
    for i in range(len(prediction)):
        if prediction[i] == ground_truth[i]:
            accuracy += 1
 
    return accuracy / len(prediction)
