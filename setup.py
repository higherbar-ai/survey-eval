#  Copyright (c) 2024 Higher Bar AI, PBC
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

from setuptools import setup

with open('README.rst') as file:
    readme = file.read()

setup(
    name='surveyeval',
    version='0.1.27',
    packages=['surveyeval'],
    python_requires='>=3.10',
    install_requires=[
        'pydantic',
        'overrides>=7.3.1,<8.0.0',
        'py-ai-workflows>=0.20.0,<1.0.0'
    ],
    extras_require={
        'parser': [
            'openpyxl>=3.0.9,<4.0.0',
            'py-ai-workflows[docs]>=0.20.0,<1.0.0'
        ]
    },
    package_data={
        'surveyeval': ['resources/*'], # include resource files in package
    },
    package_dir={'': 'src'},
    url='https://github.com/higherbar-ai/survey-eval',
    project_urls={'Documentation': 'https://surveyeval.readthedocs.io/'},
    license='Apache 2.0',
    author='Christopher Robert',
    author_email='crobert@higherbar.ai',
    description='A toolkit for survey evaluation',
    long_description=readme
)
