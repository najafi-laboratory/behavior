"""
Direct Python port of assignopts.m

Assign optional arguments (Python equivalent of MATLAB's assignopts).
"""

import inspect
from typing import Dict, Any, List, Union


def assign_opts(local_vars: Dict[str, Any], 
               kwargs: Dict[str, Any], 
               ignore_case: bool = False, 
               exact: bool = False) -> Dict[str, Any]:
    """
    Assign optional arguments from kwargs to local variables.

    This function provides functionality similar to MATLAB's assignopts,
    but adapted for Python's keyword argument system.

    INPUTS:
    local_vars   - dictionary of local variables (typically from locals())
    kwargs       - keyword arguments dictionary (typically **kwargs)
    ignore_case  - if True, perform case-insensitive matching (default: False)
    exact        - if True, require exact string matches (default: False)

    OUTPUTS:
    remain       - dictionary of unmatched keyword arguments

    USAGE:
    In a function that accepts **kwargs:
        def my_function(x, y, **kwargs):
            # Set default values
            z = 0
            verbose = False
            
            # Update with any provided kwargs
            extra_opts = assign_opts(locals(), kwargs)
            
            # Now z and verbose may have been updated from kwargs
            # extra_opts contains any unrecognized options

    @ 2009 Python port by GitHub Copilot
    Original MATLAB code by Maneesh Sahani
    """
    
    # Get the list of available option names from local_vars
    # Exclude special variables and the kwargs itself
    exclude_vars = {'kwargs', 'local_vars', 'ignore_case', 'exact', 
                   'extra_opts', 'assign_opts'}
    opts = [name for name in local_vars.keys() 
            if not name.startswith('_') and name not in exclude_vars]
    
    remain = {}
    
    # Get the caller's frame to modify variables
    frame = inspect.currentframe().f_back
    
    for key, value in kwargs.items():
        opt_key = key.lower() if ignore_case else key
        
        # Create comparison list
        if ignore_case:
            opts_compare = [opt.lower() for opt in opts]
        else:
            opts_compare = opts
        
        # Look for matches
        if exact:
            # Exact match only
            matches = [i for i, opt in enumerate(opts_compare) if opt == opt_key]
        else:
            # Prefix match (like MATLAB's strmatch)
            matches = [i for i, opt in enumerate(opts_compare) if opt.startswith(opt_key)]
        
        # If more than one match, try for exact match
        if len(matches) > 1:
            exact_matches = [i for i, opt in enumerate(opts_compare) if opt == opt_key]
            if len(exact_matches) == 1:
                matches = exact_matches
        
        # If we found a unique match, assign the value
        if len(matches) == 1:
            original_opt_name = opts[matches[0]]
            frame.f_locals[original_opt_name] = value
        else:
            # No unique match found, add to remainder
            remain[key] = value
    
    return remain


def assign_opts_simple(local_vars: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Simplified version of assign_opts for common use cases.
    
    This version only does exact, case-sensitive matching and is more
    straightforward to use in most scenarios.
    
    INPUTS:
    local_vars   - dictionary of local variables (typically from locals())
    kwargs       - keyword arguments dictionary (typically **kwargs)
    
    OUTPUTS:
    remain       - dictionary of unmatched keyword arguments
    
    USAGE:
    def my_function(x, y, **kwargs):
        # Set default values
        z = 0
        verbose = False
        
        # Update with any provided kwargs
        extra_opts = assign_opts_simple(locals(), kwargs)
    """
    
    # Get the caller's frame to modify variables
    frame = inspect.currentframe().f_back
    
    # Exclude special variables
    exclude_vars = {'kwargs', 'local_vars', 'extra_opts', 'assign_opts_simple'}
    valid_opts = {name for name in local_vars.keys() 
                  if not name.startswith('_') and name not in exclude_vars}
    
    remain = {}
    
    for key, value in kwargs.items():
        if key in valid_opts:
            frame.f_locals[key] = value
        else:
            remain[key] = value
    
    return remain