#!/usr/bin/env python3
'''
    A function def add_matrices(mat1, mat2):
    that adds two matrices element-wise.
'''

def add_matrices(mat1, mat2):
    '''
        A function def add_matrices(mat1, mat2):
        that adds two matrices element-wise.
    '''
    if isinstance(mat1, list) and isinstance(mat2, list):
        if len(mat1) == len(mat2):
            return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
        else:
            return "None"  
    elif isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2
    return "None"
