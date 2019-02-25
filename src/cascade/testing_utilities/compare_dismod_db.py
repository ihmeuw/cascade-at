from difflib import SequenceMatcher
from sqlite3 import connect


SQLITE_TABLES = "SELECT name FROM sqlite_master WHERE type='table';"


class CompareDatabases:
    def __init__(self, db_file_a, db_file_b):
        self.conn = [connect(str(dbf)) for dbf in (db_file_a, db_file_b)]
        self.tables = [sorted([x[0] for x in connl.execute(SQLITE_TABLES)]) for connl in self.conn]

    def table_diffs(self):
        sm = SequenceMatcher()
        sm.set_seqs(*self.tables)
        return [op for op in sm.get_opcodes() if op[0] != "equal"]

    def different_tables(self):
        """Which tables are different, except for the log, which always differs."""
        common = sorted(set(self.tables[0]) & set(self.tables[1]))
        differ = set()
        for c in common:
            ops = self.record_differences(c)
            if [op for op in ops if op[0] != "equal"]:
                differ.add(c)
        return differ - {"log"}

    def record_differences(self, table_name):
        records = [sorted(connl.execute(f"select * from {table_name}")) for connl in self.conn]
        sm = SequenceMatcher()
        sm.set_seqs(*records)
        return sm.get_opcodes()

    def diff_contains(self, table_name, match):
        records = [sorted(connl.execute(f"select * from {table_name}")) for connl in self.conn]
        sm = SequenceMatcher()
        sm.set_seqs(*records)
        diff_ops = [op for op in sm.get_opcodes() if op[0] != "equal"]
        diffs = list()
        for tag, i0, i1, j0, j1 in diff_ops:
            if tag == "replace":
                diffs.append(str(records[0][i0:i1]))
                diffs.append(str(records[1][j0:j1]))
        print(f"diffs {diffs}")
        return all(match in diff for diff in diffs)
