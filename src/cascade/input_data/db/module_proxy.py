"""
This is a proxy to the database modules.
"""
import importlib


BLOCK_SHARED_FUNCTION_ACCESS = False
"""
Used to control access to the testing environment. You can't load this
with from <module> import BLOCK_SHARED_FUNCTION_ACCESS. You have to
modify the value as ``module_proxy.BLOCK_SHARED_FUNCTION_ACCESS``.
"""


class ModuleProxy:
    """
    This class acts like a module. It's meant to be imported into an init.
    This exists in order to actively turn off modules during testing.
    """
    def __init__(self, module_name):
        if not isinstance(module_name, str):
            raise ValueError(f"This accepts a module name, not the module itself.")

        self.name = module_name
        try:
            self._module = importlib.import_module(module_name)
        except ModuleNotFoundError:
            self._module = None

    def __getattr__(self, name):
        if BLOCK_SHARED_FUNCTION_ACCESS:
            raise ModuleNotFoundError(
                f"Illegal access to module {self.name}. Are you trying to use "
                f"the shared functions in a unit test?")

        return getattr(self._module, name)

    def __dir__(self):
        return dir(self._module)
