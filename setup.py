from setuptools import setup, find_packages, PEP420PackageFinder

setup(
    name="cascade",
    version="0.0.1",
    packages=PEP420PackageFinder.find("src"),
    package_data={"cascade": [
        "data/*.toml",
    ]},
    package_dir={"": "src"},
    include_package_data=True,
    install_requires=["pandas", "toml"],
    extras_require={
        "testing": ["pytest", "pytest-mock", "hypothesis"],
        "ihme_databases": ["db_tools"],
    },
)
