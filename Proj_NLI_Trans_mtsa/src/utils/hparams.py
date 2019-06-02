from tensorflow.contrib.training import HParams
import os

def merge_params(p1, p2):
    params = HParams()
    v1 = p1.values()
    v2 = p2.values()
    for (k, v) in v1.items():
        params.add_hparam(k, v)
    for (k, v) in v2.items():
        params.add_hparam(k, v)
    return params


def print_params(params, sort=True, print_std=True):
    kv_list = [(k, v) for (k, v) in params.values().items()]
    if sort:
        kv_list = list(sorted(
            kv_list,
            key=lambda elem: elem[0]
        ))
    str_re = ''
    for (k, v) in kv_list:
        str_re += "%s: %s%s" % (k, v, os.linesep)
    if print_std:
        print_params(str_re)
    return str_re



def underline_to_camel(underline_format):
    camel_format = ''
    if isinstance(underline_format, str):
        for _s_ in underline_format.split('_'):
            camel_format += _s_.capitalize()
    return camel_format