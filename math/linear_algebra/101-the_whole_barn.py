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
        if len(mat1) == len(mat2):  # Check if dimensions match
            return [add_matrices(a, b) for a, b in zip(mat1, mat2)]
        else:
            return None  # Dimensions don't match
    elif isinstance(mat1, (int, float)) and isinstance(mat2, (int, float)):
        return mat1 + mat2  # Base case: add numbers
    return None  # If they don't match in type or structure

# The function will return None if the matrices have mismatched dimensions.

