from typing import Any
import random


def train_test_split(
    data: list[Any],
    test_size: float = 0.2,
    shuffle: bool = True,
    random_state: int | None = None,
) -> tuple[list[Any], list[Any]]:
    """
    Split a dataset into training and testing sets.

    Args:
        data: Input dataset.
        test_size: Fraction of data to use as the test set.
                   Must be between 0 and 1 (exclusive).
        shuffle: Whether to shuffle the data before splitting.
        random_state: Seed for reproducible shuffling.

    Returns:
        A tuple containing:
            - train_data
            - test_data

    Raises:
        ValueError: If test_size is not between 0 and 1.
    """

    if not (0 < test_size < 1):
        raise ValueError("test_size must be between 0 and 1 (exclusive).")

    data_copy = data.copy()

    if shuffle:
        rng = random.Random(random_state)
        rng.shuffle(data_copy)

    test_count = int(len(data_copy) * test_size)
    split_index = len(data_copy) - test_count

    train = data_copy[:split_index]
    test = data_copy[split_index:]

    return train, test
