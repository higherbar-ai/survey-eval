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
    version='0.1.2',
    packages=['surveyeval'],
    python_requires='>=3.10',
    install_requires=[
        'bs4~=0.0.1',
        'tiktoken~=0.5.2',
        'openai~=1.10.0',
        'langchain~=0.1.15',
        'langchain-openai~=0.0.5',
        'langchain-community~=0.0.17',
        'unstructured[local-inference]',
        'tensorboard>2.12.2',
        'uvicorn[standard]~=0.22.0',
        'pydantic~=1.10.8',
        'markdown~=3.4.3',
        'pdfkit~=1.0.0',
        'libmagic~=1.0',
        'nltk~=3.8.1',
        'spacy~=3.6.0',
        'pdfminer.six',
        'easyocr',
        'pdf2image~=1.16.3',
        'tabula-py~=2.9.0',
        'pypdf~=4.0.1',
        'pytesseract~=0.3.10',
        'tokenizers',
        'docx',
        'mammoth',
        'markdownify',
        'kor~=1.0.0',
        'scrapy~=2.11.1',
        'ipywidgets',
        'chromadb',
        'pyxform',
        'lxml',
        'requests~=2.31.0',
        'tqdm~=4.65.0',
        'overrides~=7.3.1',
    ],
    package_dir={'': 'src'},
    url='https://github.com/higherbar-ai/survey-eval',
    project_urls={'Documentation': 'https://surveyeval.readthedocs.io/'},
    license='Apache 2.0',
    author='Christopher Robert',
    author_email='crobert@higherbar.ai',
    description='A toolkit for survey evaluation',
    long_description=readme
)
