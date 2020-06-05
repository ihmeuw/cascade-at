from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade_at",
    packages=PEP420PackageFinder.find("src"),
    package_data={"cascade_at.executor": ["data/*.cfg"]},
    package_dir={"": "src"},
    use_scm_version=True,
    setup_requires=["setuptools_scm"],
    install_requires=[
        "numpy==1.17.2",
        "pandas==0.25.1",
        "scipy",
        "sqlalchemy",
        "networkx==2.3",
        "tables==3.5.2",
        "python-intervals",
        "rocketsonde @ git+https://github.com/ihmeuw/rocketsonde.git",
        "gridengineapp @ git+https://github.com/ihmeuw/gridengineapp.git",
        "h5py",
    ],
    extras_require={
        "testing": ["hypothesis", "pytest", "pytest-mock"],
        "documentation": ["sphinx", "sphinx_rtd_theme", "sphinx-autobuild", "sphinxcontrib-napoleon"],
        "ihme_databases": ["db_tools", "db_queries"],
    },
    entry_points={
        "console_scripts": [
            ["dismodel=cascade_at.executor.dismodel_main:cascade_entry"],
            ["dmchat=cascade_at.executor.chatter:chatter"],
            ["dmdummy=cascade_at.executor.chatter:dismod_dummy"],
            ["dmres2csv=cascade_at.executor.model_residuals_main:entry"],
            ["dmsr2csv=cascade_at.executor.model_results_main:entry"],
            ["dmgetsettings=cascade_at.executor.epiviz_json:entry"],
            ["dmmetrics=cascade_at.dismod.metrics:entry"],
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
