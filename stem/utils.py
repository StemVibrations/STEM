
from typing import Dict, Any, List


def merge(a :Dict[str ,Any], b :Dict[str ,Any], path:List[str]=None):
    """
    merges dictionary b into dictionary a. if existing keywords conflict it assumes
    they are concatenated in a list

    Args:
        a (Dict[str,Any]): first dictionary
        b (Dict[str,Any]): second dictionary
        path (List[str]): object to help navigate the deeper layers of the dictionary.
            Always place it as None

    Returns:
        a (Dict[str,Any]): updated dictionary with the additional dictionary `b`

    """
    if path is None:
        path = []
    for key in b:
        if key in a:
            if isinstance(a[key], dict) and isinstance(b[key], dict):
                merge(a[key], b[key], path + [str(key)])
            elif a[key] == b[key]:
                pass  # same leaf value
            elif isinstance(a[key], list) and isinstance(b[key], list):
                a[key] = a[key] + b[key]
            else:
                raise Exception("Conflict at %s" % ".".join(path + [str(key)]))
        else:
            a[key] = b[key]
    return a
