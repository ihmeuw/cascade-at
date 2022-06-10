from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade_at",
    version='0.0.1',
    description='DisMod AT Wrapper',
    package_dir={"": "src"},
    packages=PEP420PackageFinder.find("src"),
    install_requires=[
        "numpy",
        "pandas",
        "scipy",
        "hypothesis",
        "sqlalchemy",
        "dill",
        "intervaltree",
        "pytest",
        "tables",
        "networkx"
    ],
    zip_safe=False,
    extras_require={
        "ihme": [
            "db-tools",
            "db-queries",
            "gbd",
            "elmo",
            "ihme-rules",
            "jobmon",
            "cluster_utils"
        ],
        "docs": [
            "sphinx==3.2.1",
            "sphinx-rtd-theme",
            "sphinxcontrib-httpdomain",
            "sphinxcontrib-napoleon",
            "sphinx-autodoc-typehints"
        ]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Licence :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Statistics"
    ],
    entry_points={'console_scripts': [
        'configure_inputs=cascade_at.executor.configure_inputs:main',
        'dismod_db=cascade_at.executor.dismod_db:main',
        'sample=cascade_at.executor.sample:main',
        'mulcov_statistics=cascade_at.executor.mulcov_statistics:main',
        'predict=cascade_at.executor.predict:main',
        'upload=cascade_at.executor.upload:main',
        'cleanup=cascade_at.executor.cleanup:main',
        'run_cascade=cascade_at.executor.run:main',
        'run_dmdismod=cascade_at.executor.run_dmdismod:main',
        'plot=cascade_at.executor.plot:main',
        'plot_residuals=cascade_at.executor.plot_residuals:main',
        'dmdismod_script=cascade_at.fit_strategies.dismod_at_script:main',
        'dismod_ihme_input=cascade_at.executor.dismod_ihme_input:main'
        'brad_cascade=cascade_at.fit_strategies.brad_cascade_script:main'
    ]}
)
