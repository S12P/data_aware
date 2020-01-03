# Generate test and training sets
from math import floor
from random import randint


def generate_test(alpha, nbvalues, path='./data/test_set'):
    test_size = floor(alpha * nbvalues)
    test_set = set()

    with open(path, 'w') as test_file:
        while len(test_set) < test_size:
            test_value = randint(0, nbvalues)
            if test_value not in test_set:
                test_file.write('%d\n' % test_value)
            test_set.add(test_value)
    return test_set


def read_test(path='./data/test_set'):
    test_set = set()

    with open(path, 'r') as test_file:
        test_set.add(int(test_file.readline()))

    return test_set


def train_matrix_from_file(path, sep=',', test_set={}):
    """
        train_matrix_from_file computes the possibly sparse matrix described at
        the file at @path of the form

            number_of_lines, number_of_columns, number_of_values
            x, y, value
            ...
            x, y, value

        and a @test_set consisting of line numbers of the matrix describing
        file at @path that will be considered to be the test values.

        It returns:
            -the matrix (called @train_matrix)
        described by the file (where the test values are
        set to be zero)
            - a vector @test_values of the form [(x, y, value),...] consisting
            of the coordinates of the test values in the matrix and the values
            - a vector @values [(x, y),...] containing the coordinates
            and values of the non-test values (the training set) (ie the
            non-zero values of the @train_matrix)

        (',' can be replaced by any separator string by changing the @sep
        parameter)
    """

    with open(path, 'r') as fd:
        lines, columns, _ = (int(x) for x in fd.readline().split(sep))

        matrix = [[0 for _ in range(columns)] for _ in range(lines)]
        test_entries = []
        values = []

        for (n, entry) in enumerate(fd.readlines()):
            # If the entry corresponds to a training value,
            # add it to the @test_entries vector and leave matrix value
            # untouched. Else add the coordinates of the value to the
            # @values vector and set the according matrix entry to
            # the value.
            movie_id, user_id, rating = (int(x) for x in entry.split(sep))
            if n not in test_set:
                matrix[movie_id][user_id] = rating
                values.append((movie_id, user_id))
            else:
                test_entries.append((movie_id, user_id, rating))

    return matrix, test_entries, values


def train_matrix(alpha, path='./data/matrix.txt', sep=','):
    """
        @alpha: the percentage of values of the matrix used as test values
        @path: the path to the matrix describing file
        @sep: the separator used in the file at @path

        train_matrix returns a tuple (matrix, test_values, values)
        where @matrix is the matrix used to train the recommender, that
        is, the matrix described at @path minus the @test_values.
        @values contains coordinates of the non-test values.
    """

    with open(path, 'r') as fd:
        _, _, nbvalues = (int(x) for x in fd.readline().split(sep))
        test_set = generate_test(alpha, nbvalues)

    return train_matrix_from_file(path, test_set=test_set)
