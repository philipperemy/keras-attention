from setuptools import setup

setup(
    name='attention',
    version='2.2',
    description='Keras Attention Many to One',
    author='Philippe Remy',
    license='Apache 2.0',
    long_description_content_type='text/markdown',
    long_description=open('README.md').read(),
    packages=['attention'],
    # manually install tensorflow or tensorflow-gpu
    install_requires=[
        'numpy>=1.18.1',
        'keras>=2.3.1',
        'gast>=0.2.2'
    ]
)
