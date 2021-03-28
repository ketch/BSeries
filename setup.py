import setuptools
from distutils.core import setup

setup(name='BSeries',
      version='0.1',
      packages=['BSeries'],
      author=['David Ketcheson'],
      author_email='dketch@gmail.com',
      url='https://github.com/ketch/BSeries',
      description='A Package for Maniuplating Butcher Series',
      license='modified BSD',
      install_requires=['numpy','sympy','matplotlib'],
      )
