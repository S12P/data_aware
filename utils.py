def matrix_from_file(path, sep=','):
    """
        matrix_from_file computes the possibly sparse matrix described at
        the file at @path of the form

            number_of_lines, number_of_columns, number_of_values
            x, y, value
            ...
            x, y, value

        and returns it
        (',' can be replaced by any separator string by changing the @sep
        parameter)
    """

    with open(path, 'r') as fd:
        lines, columns, _ = (int(x) for x in fd.readline().split(sep))

        matrix = [[0 for _ in range(columns)] for _ in range(lines)]

        for entry in fd.readlines():
            movie_id, user_id, rating = (int(x) for x in entry.split(sep))
            matrix[movie_id][user_id] = rating

    return matrix
