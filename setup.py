from setuptools import setup, PEP420PackageFinder

setup(
    name="cascade_at",
    version='0.0.1',
    description='DisMod AT Wrapper',
    package_dir={"": "src"},
    packages=PEP420PackageFinder.find("src"),
    install_requires=[
        "numpy==1.17.2",
        "pandas==0.25.1",
        "scipy",
        "sqlalchemy",
        "tables",
        "networkx"
    ],
    zip_safe=False,
    extras_require={
        "ihme_databases": ["db_tools", "db_queries", "gbd"]
    },
    classifiers=[
        "Intended Audience :: Science/Research",
        "Licence :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Statistics"
    ]
)
