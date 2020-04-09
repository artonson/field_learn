from setuptools import setup, find_packages


def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]


# parse_requirements() returns generator of pip.req.InstallRequirement objects
REQUIREMENTS_PATH = 'requirements.txt'
install_reqs = parse_requirements(REQUIREMENTS_PATH)

setup(
    name="fieldlearn",
    version="0",
    packages=find_packages(),
    install_requires=install_reqs
)
