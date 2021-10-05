#!/bin/env python
# -*- coding: utf-8 -*-

"""
from utilities import constrain_iota, constrain_rho, call
"""
from cascade_at_gma.drill_no_csv import paths
import sys
import os
import fcntl
import tempfile
import collections
from cascade_at_gma.lib.constants import sex_name2covariate

def df_combinations(df, slice_vars, sort_vars=None):
    "Produces a generator of slices of a dataframe that have common values for the dependant variables and x_covs"
    gen = (v for k,v in df.groupby(slice_vars, as_index=False))
    if sort_vars is None:
        return gen
    else:
        return (_.sort(list(sort_vars), axis=0) for _ in gen)

def startswith_filter(startswith, strings):
    def tst(c):
        return c.startswith(startswith)
    return list(filter(tst, strings))

class dict_obj(dict):
    "A dict with object-like getters and setters."
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__

def tuple_or_None(arg):
    if arg is None: return arg
    return tuple(arg)

def list_or_None(arg):
    if arg is None: return arg
    return list(arg)

def flatten(l):
    "Flatten a nested sequence into the elements of the sequence."
    for el in l:
        if isinstance(el, collections.Iterable) and not isinstance(el, str):
            for sub in flatten(el):
                yield sub
        else:
            yield el

# Sex translations -- IHME uses both = 3, Dismod/Cascade use both = 0
sex_id2ihme_id = {1 : 1, 2 : 2, 0 : 3}
ihme_id2sex_id = {1 : 1, 2 : 2, 3 : 0}

def sex_name2sex(sex_name):
    return sex_name2covariate[sex_name]
def sex2sex_name(sex):
    if sex > 0.25: return 'male'
    if sex < -0.25: return 'female'
    return 'both'

#   IHME translations
ihme_id2sex_dict = {1 : 0.5, 2 : -0.5, 3: 0.0}
def ihme_id2sex(sex):
    return ihme_id2sex_dict[sex]
def sex2ihme_id(sex):
    if sex > 0.25: return 1
    if sex < -0.25: return 2
    return 3

#   Dismod/Cascade translations
sex_id2sex_dict = {0 : 0.0, 1 : 0.5, 2 : -0.5}
def sex_id2sex(sex):
    return sex_id2sex_dict[sex]
def sex2sex_id(sex):
    if sex > 0.25: return 1
    if sex < -0.25: return 2
    return 0

def int_or_float(_str):
    """
    int_or_float(2) # => 2
    int_or_float('2') # => 2
    int_or_float('2.0') # => 2
    int_or_float(2.0) # => 2
    int_or_float(2.1) # => 2.1
    int_or_float('2.1') # => 2.1
    """
    _str = float(_str)
    if _str % 1 == 0:
        return int(_str)
    return _str

def force_tuple(iterable_or_not):
    """
    Return a tuple of an iterable or constant argument
    Examples:
    force_tuple(np.asarray(range(3))) => (1,2,3)
    force_tuple(3) => (3,)
    force_tuple('abc') => ('abc',)
    """ 
    if isinstance(iterable_or_not, str):
        return tuple([iterable_or_not])
    try:
        return tuple(iterable_or_not)
    except:
        return tuple([iterable_or_not])

def _constrain_rate(rate_zero, rate_positive, lower, mean, upper, eps):
    if rate_zero:
        return dict(lower = 0, mean = 0, upper = 0)
    elif rate_positive:
        return dict(lower = max(eps, lower), mean = min(max(eps, lower, mean), 1, upper), upper = min(1, upper))
    else:
        return dict(lower = max(0, lower), mean = min(max(lower, mean), 1, upper), upper = min(1, upper))

def constrain_iota(rate_info, **kwds):
    return _constrain_rate(rate_zero = 'iota_zero' in rate_info,
                          rate_positive = 'iota_pos' in rate_info,
                          **kwds)

def constrain_rho(rate_info, **kwds):
    return _constrain_rate(rate_zero = 'rho_zero' in rate_info,
                          rate_positive = 'rho_pos' in rate_info,
                          **kwds)

class CalledProcessErrors(Exception): pass
def system(cmd, print_p=True, break_on_error=False):
    """
    Run the cmd like os.system does, collecting stdout and stderr. Wrap warning and error messages with an identifying string.
    If dismod_at returns a nonzero status, or the dismod_at stderr stream contains an error message, then this code raises
    a CalledProcessErrors exception.
    """
    from subprocess import Popen, PIPE, CalledProcessError
    import time

    def setNonBlocking(fd):
        """
        Set the file description of the given file descriptor to non-blocking.
        """
        flags = fcntl.fcntl(fd, fcntl.F_GETFL)
        flags = flags | os.O_NONBLOCK
        fcntl.fcntl(fd, fcntl.F_SETFL, flags)

    def Popen_nonblocking(cmd_list, print_p=True):
        # The next line fails on large stderr output from the subprocess.
        # p = Popen(cmd_list, stdin = PIPE, stdout = PIPE, stderr = PIPE, bufsize = 1, shell=True)
        # See https://stackoverflow.com/questions/1180606/using-subprocess-popen-for-process-with-large-output
        stderr_file = tempfile.NamedTemporaryFile(mode='w', delete=False)
        p = Popen(cmd_list, stdin = PIPE, stdout = PIPE, stderr = stderr_file, bufsize = 1, shell=True)
        setNonBlocking(p.stdout)
        # setNonBlocking(p.stderr)
        running = True
        while running:
            try:
                running = (p.poll() == None)
                out = p.stdout.read()
                if out:
                    if type(out) is bytes:
                        out = out.decode('utf-8')
                    print (out.strip('\n'))
            except IOError as ex:
                print ('IOERROR', ex)
            time.sleep(0.5)
        stderr_file.close()
        return p.returncode, stderr_file

    cmd = list(force_tuple(cmd))
    if print_p:
        sys.stdout.write("Starting subprocess utilities.system(%s)\n" % (' '.join(cmd))); sys.stdout.flush()

    status, stderr = Popen_nonblocking(cmd, print_p)
    with open(stderr.name, 'r') as stream: 
        child_err = stream.read()
    rtn = None
    if status != 0 or child_err: 
        if type(child_err) is bytes:
            child_err = child_err.decode('utf-8')
        if any([(('warn' in row.lower()) and not ('error' in row.lower())) for row in child_err.split('\n') if len(row.strip()) > 0]):
            rtn = 'Command %s returned exit status %s, and the following warning message:\n\t"""\n\t%s"""'% (cmd, status, '\n\t'.join(child_err.split('\n')))
            print (rtn, file=sys.stderr)
        else:
            rtn = 'Command %s returned non-zero exit status %s, and the following error message:\n\t"""\n\t%s"""'% (cmd, status, '\n\t'.join(child_err.split('\n')))
            if break_on_error:
                print (rtn)
                import pdb; pdb.set_trace()
            raise CalledProcessErrors(rtn)
    # sys.stdout.write(" done.\n"); sys.stdout.flush()
    return status, rtn

if (__name__ == '__main__'):

    import unittest

    class TestUtilities(unittest.TestCase):
        def test_force_tuple(self):
            self.assertTrue(force_tuple(1) == (1,))
            self.assertTrue(force_tuple([1]) == (1,))
            self.assertTrue(force_tuple((1,)) == (1,))
            self.assertTrue(force_tuple('a') == ('a',))
            self.assertTrue(force_tuple('ab') == ('ab',))

    unittest.main(exit = False)
