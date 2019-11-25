
type_mappings = {
    'int': int,
    'str': str,
    'float': float
}


def parse_options(option_list):
    """
    Parse a KEY=VALUE=TYPE command line arg
    that comes in a list

    :param option_list: List[str]
    :return: dictionary of options that can be passed to the dismod database
    """
    d = dict()
    for o in option_list:
        o = o.split('=')
        key = o[0]
        val = o[1]
        type_func = type_mappings[o[2]]
        d.update({key: type_func(val)})
    return d

