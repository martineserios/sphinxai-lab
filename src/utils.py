# import ext libs
from functools import reduce


def head_compensation_mapper(value):
    value = abs(value)
    category = ''

    if value <= abs(4):
        category = 'PRO'
    elif (value > abs(4)) & (value <= abs(8)):
        category = 'AR'
    elif (value > abs(8)) & (value <= abs(16)):
        category = 'AM'
    elif (value > abs(16)):
        category = 'BEG'
    return category

def percent_elements_in_dict(categs_dict):
    total = sum(categs_dict.values())
    percent = {key: f'{round(100 * (value/total))}%' for key, value in categs_dict.items()}
    
    return percent


def list_mean(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)