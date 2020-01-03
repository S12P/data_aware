# Generate test and training sets
from math import randint, floor


def generate_test(alpha, nbvalues):
    test_size = floor(alpha * nbvalues)

    test_set = {}

    with open('./data/test_set') as test_file:
        while len(test_set) < test_size:
            test_value = randint(nbvalues)
            if test_value not in test_set:
                test_file.write('%d\n', test_value)
            test_set.add(test_value)


def train_matrix_from_file(path, sep=',', test_set={}):
    """
        train_matrix_from_file computes the possibly sparse matrix described at
        the file at @path of the form

            number_of_lines, number_of_columns, number_of_values
            x, y, value
            ...
            x, y, value

        and a test_set consisting of line numbers of the matrix descripting
        file at @path that will be considered to be the test values.

        It returns the matrix (called train_matrix)
        described by the file (where the test values are
        set to be zero) and the vector
        test_values of the form [(x, y, value),...] consisting of
        the coordinates of the test values in the matrix and the values.

        (',' can be replaced by any separator string by changing the @sep
        parameter)
    """

    with open(path, 'r') as fd:
        lines, columns, _ = (int(x) for x in fd.readline().split(sep))

        matrix = [[0 for _ in range(columns)] for _ in range(lines)]
        test_entries = []

        for (entry, n) in enumerate(fd.readlines()):
            # If the entry corresponds to a training value,
            # add it to the test_entries vector and leave matrix value
            # untouched
            if n not in test_set:
                movie_id, user_id, rating = (int(x) for x in entry.split(sep))
                matrix[movie_id][user_id] = rating
            else:
                test_entries.append((movie_id, user_id, rating))

    return matrix, test_entries
