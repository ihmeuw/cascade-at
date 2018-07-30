from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade",
    version="0.0.1",
    packages=PEP420PackageFinder.find("src"),
    package_data={"cascade.executor": [
        "data/*.toml",
    ]},
    package_dir={"": "src"},
    install_requires=["pandas", "toml"],
    extras_require={
        "testing": ["pytest", "pytest-mock", "hypothesis"],
        "documentation": ["sphinx", "sphinx_rtd_theme", "sphinx-autobuild",
                          "sphinxcontrib-napoleon"],
        "ihme_databases": ["db_tools"],
    },
    entry_points={
        "console_scripts": [
            ["dmchat=cascade.executor.chatter:chatter"],
            ["dmdummy=cascade.executor.chatter:dismod_dummy"],
        ]
    },
)
