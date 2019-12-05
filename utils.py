def create_set(set, sigma):

    set_return = [set[i] + (sigma[i], ) for i in range(len(set))]

    return set_return