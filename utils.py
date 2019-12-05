def create_set(set, sigma):

    set_return = [set[i] + (sigma[i], ) for i in range(len(set))]

    return set_return


def cal_index(indices, subindices):
    num = 0
    for i in range(len(indices)):
        if indices[i] == 1:
            if subindices[num] != 1:
                indices[i] = 0
            elif subindices[num] != 1:
                indices[i] = 1
            num += 1
        else:
            continue

