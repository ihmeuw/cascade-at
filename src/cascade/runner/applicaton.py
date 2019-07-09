import networkx as nx


class Application:
    """
    Responsible for management of settings and creation of job graphs.
    """
    def __init__(self):
        pass

    def add_arguments(self, parser):
        return parser

    def create_settings(self, args):
        pass

    def load_settings(self, args):
        pass

    def save_settings(self):
        pass

    def graph_of_jobs(self, _args):
        return nx.DiGraph()

    def sub_graph_to_run(self, _args):
        return nx.DiGraph()
