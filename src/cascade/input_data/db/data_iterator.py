def grouped_by_count(identifiers, count):
    for i in range((len(identifiers) - 1) // count + 1):
        yield identifiers[i * count:(i + 1) * count]
