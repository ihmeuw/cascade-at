class FileEntity:
    def __init__(self, relative_path):
        self._relative_path = relative_path

    def validate(self, base_path):
        return (base_path / self._relative_path).exists()


class Stage:
    def __init__(self):
        self._inputs = list()
        self._outputs = list()

    def __call__(self):
        pass

    @property
    def inputs(self):
        return self._inputs

    @property
    def outputs(self):
        return self._outputs
