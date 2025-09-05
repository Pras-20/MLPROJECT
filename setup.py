from setuptools import find_packages,setup 
from typing import List 

def get_requirements(filepath:str)->List[str]:
    '''
    this function will return a list of requirements that need to be installed
    '''

    requirements=[]
    with open(filepath,'r') as f:
        requirements=f.readlines()
        requirements=[r.replace("\n","") for r in requirements]
        if '-e .' in requirements:
            requirements.remove('-e .')
    return requirements

setup(
    name="MLPROJECT",
    version="0.0.1",
    author="prasanna",
    author_email="prasannakumarpk2023@gmail.com",
    package=find_packages(),
    install_requires=get_requirements("requirements.txt")
)   