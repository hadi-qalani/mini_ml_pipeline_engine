def _confusion_counts(
    y_true: list,
    y_pred: list,
) -> tuple[int, int, int, int]:
    """
    Calculate the confusion matrix counts for binary classification.

    Args:
        y_true (list[int]): The ground truth labels. Each element must be 0 or 1.
        y_pred (list[int]): The predicted labels. Each element must be 0 or 1.

    Returns:
        tuple[int, int, int, int]: A tuple containing the counts of
            true positives (TP),
            false positives (FP),
            true negatives (TN),
            and false negatives (FN),
            in that order.

    Raises:
        TypeError: If either input is not a list.
        ValueError: If the input lists are empty.
        ValueError: If the input lists have different lengths.
        ValueError: If any element is not 0 or 1.
    """
    if not isinstance(y_pred, list) or not isinstance(y_true, list):
        raise TypeError("either input is not a list")
    if len(y_true) != len(y_pred):
        raise ValueError("length of lists are not equal")
    if not y_true or not y_pred:
        raise ValueError("lists should not be empty")

    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(len(y_true)):
        if not (isinstance(y_pred[i], (int,)) and isinstance(y_true[i], (int,))):
            raise TypeError("list elements are not int")
        if y_pred[i] not in [0, 1] or y_true[i] not in [0, 1]:
            raise ValueError("an element is not 0 or 1")

    for i in range(len(y_true)):
        if y_pred[i] == 1 and y_true[i] == 1:
            TP += 1
        elif y_pred[i] == 0 and y_true[i] == 0:
            TN += 1
        elif y_pred[i] == 1 and y_true[i] == 0:
            FP += 1
        else:
            FN += 1
    return (TP, FP, TN, FN)


def accuracy_score(
    y_true: list[int],
    y_pred: list[int],
) -> float:
    """
    Calculates the accuracy score for binary classification.

    Args:
        y_true (list[int]): Ground truth labels. Each element must be 0 or 1.
        y_pred (list[int]): Predicted labels. Each element must be 0 or 1.

    Returns:
        float: The accuracy score, computed as the ratio of correctly
            classified samples to the total number of samples.

    Raises:
        TypeError: If either input is not a list or contains non-integer elements.
        ValueError: If the input lists are empty, have different lengths,
            or contain values other than 0 or 1.
    """

    TP, _, TN, _ = _confusion_counts(y_true, y_pred)

    return (TP + TN) / len(y_true)


def precision_score(
    y_true: list,
    y_pred: list,
) -> float:
    """Calculate the precision score.

    Precision is defined as TP / (TP + FP). If there are no predicted
    positive samples (TP + FP == 0), this function returns 0.0.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        The precision score.

    Raises:
        ValueError: If the input lists are empty or have different lengths.
    """
    TP, FP, _, _ = _confusion_counts(y_true, y_pred)

    if TP + FP == 0:
        return 0.0

    return TP / (TP + FP)


def recall_score(y_true: list, y_pred: list) -> float:
    """Calculate the recall score.

    Recall is defined as TP / (TP + FN). If there are no actual
    positive samples (TP + FN == 0), this function returns 0.0.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        The recall score.

    Raises:
        ValueError: If the input lists are empty or have different lengths.
    """
    TP, _, _, FN = _confusion_counts(y_true, y_pred)

    if TP + FN == 0:
        return 0.0

    return TP / (TP + FN)


def f1_score(y_true: list, y_pred: list) -> float:
    """Calculate the F1 score.

    The F1 score is the harmonic mean of precision and recall.
    If the sum of precision and recall is 0, this function returns 0.0.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted binary labels.

    Returns:
        The F1 score.

    Raises:
        ValueError: If the input lists are empty or have different lengths.
    """
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)

    if precision + recall == 0:
        return 0.0

    return 2 * precision * recall / (precision + recall)
