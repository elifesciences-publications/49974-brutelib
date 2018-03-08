from setuptools import setup

setup(
    name='brutelib',
    version='1.0',
    packages=['brutelib'],
    url='',
    license='',
    author='Paul Pfeiffer',
    author_email='pfeifferpaul90@gmail.com',
    description='Brute force simulation of a special stochastic hybrid system.',
    install_requires=[
          'dill', 'h5py', 'numpy', 'scipy', 'tables', 'tqdm',
      ],
)
