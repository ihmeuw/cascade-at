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
            "jobmon==2.0.3",
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
        "Programming Language :: Python :: 3.8",
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
        'run_dmdismod=cascade_at.executor.run_dmdismod:main'
    ]}
)
