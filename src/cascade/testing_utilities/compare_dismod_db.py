from difflib import SequenceMatcher
from sqlite3 import connect


SQLITE_TABLES = "SELECT name FROM sqlite_master WHERE type='table';"


def compare(db_file_a, db_file_b):
    connections = [connect(str(dbf)) for dbf in (db_file_a, db_file_b)]
    diffs, equal_tables = table_differences(*connections)
    for t in equal_tables:
        record_diff = record_differences(t, *connections)


def table_differences(conna, connb):
    table_list = [sorted([x[0] for x in connl.execute(SQLITE_TABLES)]) for connl in (conna, connb)]
    sm = SequenceMatcher()
    sm.set_seqs(*table_list)
    return sm.get_opcodes(), sorted(set(table_list[0]) & set(table_list[1]))


def record_differences(table_name, conna, connb):
    records = [sorted(connl.execute(f"select * from {table_name}")) for connl in (conna, connb)]
    sm = SequenceMatcher()
    sm.set_seqs(*records)
    return sm.get_opcodes()
