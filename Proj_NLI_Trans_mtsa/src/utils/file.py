import os, pickle, json
from os.path import dirname, basename


def mp_join(*args):
    dir_path = os.path.join(*args)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def save_file(data, file_path, data_name='data', mode='pickle'):
    if mode == 'pickle':
        with open(file_path, 'wb') as f:
            pickle.dump(obj=data,
                        file=f)
    elif mode == 'json':
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(obj=data,
                      fp=f)
    else:
        raise(ValueError,'Function save_file does not have mode %s' % (mode))


def load_file(file_path, data_name='data', mode='pickle'):
    data = None
    if os.path.isfile(file_path):
        if mode == 'pickle':
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
        elif mode == 'json':
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise (ValueError, 'Function save_file does not have mode %s' % (mode))

    else:
        print('Have not found the file')

    return data


