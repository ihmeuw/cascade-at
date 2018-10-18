"""
This script is for testing how we run DismodAT.
It sleeps and sends messages.
"""
import os
import random
import sys
import time

from cascade.core.log import getLoggers
CODELOG, MATHLOG = getLoggers(__name__)


def chatter():
    """
    This is the ``dmchat`` script.
    """
    talk_cnt = int(sys.argv[1])
    return_code = int(sys.argv[2])
    death = int(sys.argv[3])

    if death > 0:
        death_idx = random.randrange(talk_cnt)
    else:
        death_idx = talk_cnt

    for talk_idx in range(talk_cnt):
        sys.stdout.write("out" + os.linesep)
        sys.stderr.write("err" + os.linesep)
        if talk_idx == death_idx:
            os.kill(os.getpid(), death)
        time.sleep(1)

    sys.exit(return_code)


def dismod_dummy():
    """
    This is the ``dmdummy`` script.
    Dismod_at commands look like::

        dismod_at <database_name> command <arg1> <arg2>

    """
    dm_file = sys.argv[1]

    with open(dm_file, "a") as dm_stream:
        dm_stream.write(" ".join(sys.argv) + os.linesep)


if __name__ == "__main__":
    chatter()
