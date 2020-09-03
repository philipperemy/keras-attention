from setuptools import setup

from attention import VERSION

setup(
    name='attention',
    version=VERSION,
    description='Keras Simple Attention',
    author='Philippe Remy',
    license='Apache 2.0',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['attention'],
    install_requires=[
        'numpy>=1.18.1',
        'keras>=2.3.1',
        'tensorflow>=2.1'
    ]
)
