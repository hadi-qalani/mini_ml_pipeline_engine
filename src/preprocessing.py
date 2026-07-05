from math import sqrt


def _validation(data: list[float]) -> None:
    if type(data) is not list:
        raise TypeError("you should insert a list data type")

    if not data:
        raise ValueError("data should not be empty")
    for num in data:
        if not isinstance(num, (int, float)):
            raise TypeError("all elements must be int or float")


def mean(data: list[float]) -> float:
    """
        calculate and return the arithmetic mean of a list of numbers.

    Args:
        data: A non-empty list of floating-point numbers.

    Returns:
        The arithmetic mean of the input values.

    Raises:
        ValueError: If the input list is empty.
    """

    _validation(data)

    sum_list = 0
    for i in range(0, len(data)):
        sum_list += data[i]

    return sum_list / len(data)


def std(data: list[float]) -> float:
    """
    Calculate the population standard deviation of a list of numbers.
    Args: data (list[float]): A non-empty list of numeric values.
    Returns: float: The population standard deviation of the input data.
    Raises:
    ValueError: If data is not a list. ValueError: If data is empty.
    ValueError: If any element in data is not numeric.
    """

    _validation(data)

    mean_data = mean(data)
    sum_for_std = 0
    for num in data:
        sum_for_std += (num - mean_data) ** 2

    return sqrt((sum_for_std / len(data)))


def min_max_normalize(data: list[float]) -> list[float]:
    """
    Normalize values to the range [0, 1].
    """
    normal_data = []
    _validation(data)

    max_num = max(data)
    min_num = min(data)
    if max_num == min_num:
        raise ValueError("Cannot normalize because all values are identical.")
    for num in data:
        normalized = (num - min_num) / (max_num - min_num)
        normal_data.append(normalized)

    return normal_data


def z_score_normalize(data: list[float]) -> list[float]:
    """
    Normalize a list of numbers using the Z-score normalization method.
    Args: data (list[float]): A list of numeric values to normalize.
    Returns: list[float]: A new list containing the normalized values.
    Raises:
    TypeError: If `data` is not a list. ValueError: If `data` is empty.
    ValueError: If any element in `data` is not an integer or float.
    ValueError: If the standard deviation is zero.
    """
    _validation(data)

    mean_data = mean(data)
    std_data = std(data)
    if std_data == 0:
        raise ValueError("std in equal to 0 and z-score couldn't be computed")
    z_normalized_data = []

    for num in data:
        z_normalized_data.append((num - mean_data) / std_data)
    return z_normalized_data
