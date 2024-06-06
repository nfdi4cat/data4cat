from setuptools import setup

setup(
    name='data4cat',
    version='0.1.0',
    description='Python library with datsets from NFDI4Cat',
    url='',
    author='Stefan Palkovits et. al',
    license= 'BSD 2-clause',
    packages=['data4cat'],
    install_requires=[ 
        'numpy',
        'matplotlib',
        'scikit-learn',
        'pandas',
        
    ],

    classifiers= [
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Science/Research',
        'Operating System :: Windows',
        'Programming Language :: Python :: 3'
        ],
)