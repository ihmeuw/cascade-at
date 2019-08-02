def grouped_by_count(identifiers, count):
    """
    Given a list, iterates over that list in sets of size count.
    The last set will be of size less than or equal to count.

    Args:
        identifiers (List): Can be any iterable.
        count (int): Number of items to return on each iteration.

    Returns:
        List: On each iteration, returns a list of count members
        or, on the last iteration, less-than-or-equal-to count members.
    """
    identifiers = list(identifiers)
    for i in range((len(identifiers) - 1) // count + 1):
        yield identifiers[i * count:(i + 1) * count]
