# import ext libs
from functools import reduce


def percent_elements_in_dict(categs_dict):
    total = sum(categs_dict.values())
    percent = {key: f'{round(100 * (value/total))}%' for key, value in categs_dict.items()}
    
    return percent


def list_mean(lst):
    return reduce(lambda a, b: a + b, lst) / len(lst)