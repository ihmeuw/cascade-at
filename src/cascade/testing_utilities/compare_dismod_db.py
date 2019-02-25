from difflib import SequenceMatcher
from sqlite3 import connect


SQLITE_TABLES = "SELECT name FROM sqlite_master WHERE type='table';"


class CompareDatabases:
    def __init__(self, db_file_a, db_file_b):
        self.conn = [connect(str(dbf)) for dbf in (db_file_a, db_file_b)]

    def table_diffs(self):
        table_list = [sorted([x[0] for x in connl.execute(SQLITE_TABLES)]) for connl in self.conn]
        sm = SequenceMatcher()
        sm.set_seqs(*table_list)
        return sm.get_opcodes(), sorted(set(table_list[0]) & set(table_list[1]))


def record_differences(table_name, conna, connb):
    records = [sorted(connl.execute(f"select * from {table_name}")) for connl in (conna, connb)]
    sm = SequenceMatcher()
    sm.set_seqs(*records)
    return sm.get_opcodes()

# opcodes are replace, delete, insert, equal
# (tag, i0, i1, j0, j1). How do we interpret replace?
