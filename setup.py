# import ez_setup
# ez_setup.use_setuptools()

from setuptools import setup, find_packages


__package__ = 'mie_perso'
__version__ = '1.0.0'

setup(
    name=__package__,
    version=__version__,
    packages=find_packages(exclude=['build']),
    package_data={
         '': ['*.nc','*.txt','*.csv','*.dat'],
        # 'aux': ['data/aux/*']
    },
    include_package_data=True,

    url='',
    license='MIT',
    author='T. Harmel',
    author_email='tristan.harmel@gmail.com',
    description='simple code calling PyMieScatt for multiprocessor computation',

    # Dependent packages (distributions)
    install_requires=['numpy','scipy','pandas','xarray','lmfit',
                      'matplotlib','PyMieScatt'],


    entry_points={
          'console_scripts': [
              'mie_perso = TODO'
          ]}
)
