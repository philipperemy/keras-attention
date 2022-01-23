from setuptools import setup, find_packages

setup(
    name='attention',
    version='4.1',
    description='Keras Simple Attention',
    author='Philippe Remy',
    license='Apache 2.0',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.1',
        'tensorflow>=2.1'
    ]
)
