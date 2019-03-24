"""
The goal of this script is to show what hosts this application contacts.
First, you run the app using strace::

    strace  -f -e trace=network dmcascade --mvid 265976 --db-only z.db > strace.txt

Then this app will read the strace, search for internet addresses, and find
the dotted-names for them.
"""
from pathlib import Path
import re
import subprocess


ADDR_SEARCH = re.compile(r'inet_addr\("([\d\.]+)')
NAME_SEARCH = re.compile(r"name = ([a-z0-9-\.]+)")


def find_addresses(file_path):
    """
    Args:
        file_path (Path): Path to a file with stderr from strace of network.
    """
    address = set()
    for l in file_path.open().readlines():
        addr_match = ADDR_SEARCH.search(l)
        if addr_match:
            address.add(addr_match.group(1))
    return address


def lookup(addr):
    result = subprocess.check_output(["nslookup", addr], universal_newlines=True)
    name_match = NAME_SEARCH.search(result)
    # print(f"============\n{result}\n===========")
    if name_match:
        # Get rid of period at end of nslookup result with [:-1].
        return addr, name_match.groups()[0][:-1]
    else:
        return addr, addr


if __name__ == "__main__":
    addr = find_addresses(Path("strace.txt"))
    res = [lookup(one_addr) for one_addr in addr]
    print("\n".join([f"{y}: {x}" for (x, y) in sorted(res)]))
