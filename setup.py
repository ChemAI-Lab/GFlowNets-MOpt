from setuptools import setup, find_packages

setup(
    name='gflow_vqe',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[],  # list dependencies here, e.g., ['numpy']
    author='Isaac Huidobro',
    author_email='huidobri@mcmaster.ca',
    description='Grouping using GFlowNets',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
    ],
)
