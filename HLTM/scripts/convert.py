#!/usr/bin/env python3

import re
import gzip, bz2
    

def convert_to_sparse(filename, lists):
    
    def to_attr(index):
        return f'c{index:03d}'

    print(f'writing to {filename}', end='...')
    with gzip.open(filename, 'wt') as fout:
        for i in range(len(lists)):
            for v in lists[i]:
                fout.write(f'{i}, {to_attr(v)}\n')

            if i < 1000000 and i % 100000 == 0:
                print(i, end='...')
            elif i % 1000000 == 0:
                print(i, end='...')

        print('done')
        
def read_lists(filename):
    import numpy as np

    print(f'Opening {filename}')
    if filename.endswith('.gz'):
        f = gzip.GzipFile(filename, 'r')
    elif filename.endswith('.bz2'):
        f = bz2.Bz2File(filename, 'r')
    else:
        f = filename

    print(f'Loading {filename}')
    return np.load(f, allow_pickle=True)

def convert_npy_file(filename):
    sparse_filename = re.sub('.npy(.bz2|.gz)?$', '.sparse.txt.gz', filename)
    # sparse_filename = filename.replace('.npy', '.sparse.txt')
    convert_to_sparse(sparse_filename, read_lists(filename))

def convert_from_npy_to_sparse(name):
    convert_to_sparse(f'{name}.sparse.txt', read_lists(f'{name}.npy'))
    

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('convert npy_file [...]')
    else:
        for f in sys.argv[1:]:
            convert_npy_file(f)


        
#    convert_from_npy_to_sparse('list_95_whole_1')
#    convert_from_npy_to_sparse('list_95_whole')


