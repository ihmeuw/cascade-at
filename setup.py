from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade",
    packages=PEP420PackageFinder.find("src"),
    package_data={"cascade.executor": ["data/*.toml"]},
    package_dir={"": "src"},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=["numpy", "pandas", "scipy", "toml", "sqlalchemy", "networkx",
                      "tables", "python-intervals", "rocketsonde @ git+https://github.com/ihmeuw/rocketsonde.git",
                      "h5py"],
    extras_require={
        "testing": ["hypothesis", "pytest", "pytest-mock"],
        "documentation": ["sphinx", "sphinx_rtd_theme", "sphinx-autobuild", "sphinxcontrib-napoleon"],
        "ihme_databases": ["db_tools", "db_queries", "save_results"],
    },
    entry_points={
        "console_scripts": [
            ["dmcascade=cascade.executor.epiviz_runner:entry"],
            ["dmchat=cascade.executor.chatter:chatter"],
            ["dmdummy=cascade.executor.chatter:dismod_dummy"],
            ["dmcsv2db=cascade.executor.no_covariate_main:entry"],
            ["dmres2csv=cascade.executor.model_residuals_main:entry"],
            ["dmsr2csv=cascade.executor.model_results_main:entry"],
            ["dmgetsettings=cascade.executor.epiviz_json:entry"],
        ]
    },
    scripts=["scripts/dmdismod", "scripts/dmdismodpy"],
    zip_safe=False,
    classifiers=[
        "Intended Audience :: Science/Research",
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Statistics",
    ],
)
