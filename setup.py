from setuptools import find_packages
from setuptools import setup

with open("requirements.txt") as f:
    content = f.readlines()
requirements = [x.strip() for x in content if "git+" not in x]

setup(name='wfa',
      version="0.0.4",
      description="many bug fixes and small changes",
      license="SE",
      author="SE",
      author_email="pabloknecht@gmail.com",
      url="https://github.com/pabloknecht/watching-from-above",
      install_requires=requirements,
      packages=find_packages(),
      # include_package_data: to install data from MANIFEST.in
      include_package_data=True,
      zip_safe=False)
