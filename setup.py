import setuptools
from distutils.core import setup

# Use README as description on Pypi
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='BSeries',
      version='0.1',
      packages=['BSeries'],
      author=['David Ketcheson'],
      author_email='dketch@gmail.com',
      url='https://github.com/ketch/BSeries',
      description='A Package for Maniuplating Butcher Series',
      license='modified BSD',
      install_requires=['numpy','sympy','matplotlib'],
      long_description=long_description,
      long_description_content_type='text/markdown'
      )
