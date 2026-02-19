"""
Direct Python port of parseFilename.m

Extracts method, xDim and cross-validation fold from results filename.
"""

import re
from typing import Dict, Tuple, Any, Optional


def parse_filename(filename: str) -> Dict[str, Any]:
    """
    Extracts method, xDim and cross-validation fold from results filename,
    where filename has format [method]_xDim[xDim]_cv[cvf].mat or .pkl.

    INPUTS:
    filename - filename string to parse

    OUTPUTS:
    result   - dictionary with fields 'method', 'x_dim', and 'cvf'
               Returns empty dict if parsing fails

    @ 2009 Byron Yu -- byronyu@stanford.edu
    Python port by GitHub Copilot
    """
    
    result = {}
    
    # Find underscores
    undi = [i for i, char in enumerate(filename) if char == '_']
    if not undi:
        return result
    
    # Extract method (everything before first underscore)
    result['method'] = filename[:undi[0]]
    
    # Extract the remaining part after first underscore
    remaining = filename[undi[0] + 1:]
    
    # Use regex to parse xDim and cv parts
    # Pattern matches: xDim<number>_cv<number>.mat or xDim<number>_cv<number>.pkl
    # Also matches: xDim<number>.mat or xDim<number>.pkl (no cv part)
    pattern = r'xDim(\d+)(?:_cv(\d+))?\.(?:mat|pkl)'
    match = re.match(pattern, remaining)
    
    if not match:
        return {}
    
    # Extract xDim
    result['x_dim'] = int(match.group(1))
    
    # Extract cvf (cross-validation fold)
    if match.group(2) is not None:
        result['cvf'] = int(match.group(2))
    else:
        result['cvf'] = 0
    
    return result


def parse_filename_with_error(filename: str) -> Tuple[Dict[str, Any], bool]:
    """
    Alternative version that returns error flag like the original MATLAB function.

    INPUTS:
    filename - filename string to parse

    OUTPUTS:
    result   - dictionary with fields 'method', 'x_dim', and 'cvf'
    err      - boolean that indicates if input string is invalid filename

    @ 2009 Byron Yu -- byronyu@stanford.edu
    Python port by GitHub Copilot
    """
    
    result = {}
    err = False
    
    # Find underscores
    undi = [i for i, char in enumerate(filename) if char == '_']
    if not undi:
        err = True
        return result, err
    
    # Extract method (everything before first underscore)
    result['method'] = filename[:undi[0]]
    
    # Extract the remaining part after first underscore
    remaining = filename[undi[0] + 1:]
    
    # Use regex to parse xDim and cv parts
    pattern = r'xDim(\d+)(?:_cv(\d+))?\.(?:mat|pkl)'
    match = re.match(pattern, remaining)
    
    if not match:
        err = True
        return result, err
    
    # Extract xDim
    result['x_dim'] = int(match.group(1))
    
    # Extract cvf (cross-validation fold)
    if match.group(2) is not None:
        result['cvf'] = int(match.group(2))
    else:
        result['cvf'] = 0
    
    return result, err