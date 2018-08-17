from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade",
    version="0.0.1",
    packages=PEP420PackageFinder.find("src"),
    package_data={"cascade.executor": ["data/*.toml"]},
    package_dir={"": "src"},
    install_requires=["numpy", "pandas", "scipy", "toml", "sqlalchemy", "networkx"],
    extras_require={
        "testing": ["hypothesis", "pytest", "pytest-mock"],
        "documentation": ["sphinx", "sphinx_rtd_theme", "sphinx-autobuild", "sphinxcontrib-napoleon"],
        "ihme_databases": ["db_tools", "db_queries", "save_results"],
    },
    entry_points={
        "console_scripts": [
            ["dmchat=cascade.executor.chatter:chatter"],
            ["dmdummy=cascade.executor.chatter:dismod_dummy"],
            ["dmcsv2db=cascade.executor.no_covariate_main:entry"],
        ]
    },
    scripts=["scripts/dmdismod", "scripts/dmdismodpy"],
    zip_safe=False,
    classifiers=[
        "Intendend Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Statistics",
    ],
)
