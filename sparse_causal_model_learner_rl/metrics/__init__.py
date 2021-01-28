def find_key(dct, key_substr):
    """Get a key from a dictionary containing a substring."""
    keys = dct.keys()
    keys_match = [k for k in keys if key_substr in k]
    if len(keys_match) > 1:
        keys_match = keys_match[:1]
    assert len(keys_match) == 1, f"Unknown substring {key_substr} in {keys}, appears {len(keys_match)} times"
    return keys_match[0]


def find_value(dct, key_substr):
    """Get a value from a dict given a substring in a key."""
    return dct[find_key(dct, key_substr)]
