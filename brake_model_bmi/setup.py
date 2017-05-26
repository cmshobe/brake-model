from setuptools import setup, find_packages

setup(name='BRaKE',
      version='1.0',
      author='Charles Shobe',
      author_email='charles.shobe@colorado.edu',
      license='GNU GPLv3',
      description='Blocky River and Knickpoint Evolution Model',
      install_requires=('numpy', 'pyyaml', 'scipy'),
      packages=find_packages(exclude=['*.tests']),
)