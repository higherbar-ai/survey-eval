#  Copyright (c) 2023-24 Higher Bar AI, PBC
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

"""Utility functions for reading and parsing survey files."""

import os
import tempfile
import logging
import asyncio
import requests
from urllib.parse import urlparse

from html_tools import MarkdownifyHTMLProcessor
from langchain_core.pydantic_v1 import BaseModel, Field
from typing import List, Optional, TypedDict, Tuple
import uuid
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain.chains import LLMChain
from langchain.callbacks import get_openai_callback
from typing import Callable
import nltk
import spacy
import csv
import re
import xml.etree.ElementTree as ElementTree
import pytesseract
from pypdf import PdfReader
from tabula.io import read_pdf
from pdf2image import convert_from_path
from langchain.schema import Document as SchemaDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader
import importlib.resources as pkg_resources

# initialize global resources
parser_logger = logging.getLogger(__name__)
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
empty_form_path = pkg_resources.files('surveyeval').joinpath('resources/EmptyForm.xlsx')

# initialize global variables
langchain_splitter_chunk_size = 6000
langchain_splitter_overlap_size = 500

# initialize parsing schema
module_spec = ('Represents the main section or category within which a question is '
               'located (e.g., "Health" or "Demographics"). It might include a number or index, but should '
               'also include a short title. All questions within a module should have the same module.')
question_id_spec = ('A numeric or alphanumeric identifier or short variable name identifying a '
                    'specific question, usually located just before or at the beginning of a new question.')
question_spec = ('A single question or label/description of a single form field, often following '
                 'a numerical code or identifier like "2.01." or "gender:" Must be text designed to elicit '
                 'specific information, often in the form of a question (e.g., "How old are you?") or prompt '
                 '(e.g., "Your age:"). Might be in different languages, but the structure remains the same.')
instructions_spec = ('Instructions or other guidance about how to ask or answer the '
                     'question, including enumerator or interviewer instructions. If the question includes '
                     'a list of specific response options, do NOT include those in the instructions.')
options_spec = ("The list of specific response options for multiple-choice questions, "
                "including both the label for the option and the internal value, if specified. For example, "
                "a 'Male' label might be coupled with an internal value of '1', 'M', or even 'male'. "
                "Separate response options with a space, three pipe symbols ('|||'), and another space, and, "
                "if there is an internal value, add a space, three # symbols ('###'), and the internal value "
                "at the end of the label. For example: 'Male ### 1 ||| Female ### 2' (codes included) or "
                "'Male ||| Female' (no codes); 'Yes ### yes ||| No ### no', 'Yes ### 1 ||| No ### 0', 'Yes ### "
                "y ||| No ### n', or 'YES ||| NO'.")
language_spec = 'The primary language in which the question is written.'


class Question(BaseModel):
    """Information about a question."""

    module: Optional[str] = Field(..., description=module_spec)
    question_id: Optional[str] = Field(..., description=question_id_spec)
    question: Optional[str] = Field(..., description=question_spec)
    instructions: Optional[str] = Field(..., description=instructions_spec)
    options: Optional[str] = Field(..., description=options_spec)
    language: Optional[str] = Field(..., description=language_spec)


class QuestionList(BaseModel):
    """List of extracted questions from a questionnaire."""

    questions: List[Question]


class Example(TypedDict):
    """
    A representation of an example consisting of text input and expected tool calls.

    For extraction, the tool calls are represented as instances of pydantic model.
    """

    input: str                          # the example text
    tool_calls: List[BaseModel]         # instances of pydantic model that should be extracted


prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an expert questionnaire extraction algorithm. "
            "Only extract relevant information from the text. "
            "If you do not know the value of an attribute asked "
            "to extract, return null for the attribute's value.",
        ),
        MessagesPlaceholder("examples"),
        ("human", "{text}"),
    ]
)


def tool_example_to_messages(example: Example) -> List[BaseMessage]:
    """
    Convert an example into a list of messages that can be fed into an LLM.

    This code is an adapter that converts our example to a list of messages
    that can be fed into a chat model.

    The list of messages per example corresponds to:

    1) HumanMessage: contains the content from which content should be extracted.
    2) AIMessage: contains the extracted information from the model
    3) ToolMessage: contains confirmation to the model that the model requested a tool correctly.

    The ToolMessage is required because some of the chat models are hyper-optimized for agents
    rather than for an extraction use case.
    """
    messages: List[BaseMessage] = [HumanMessage(content=example["input"])]
    openai_tool_calls = []
    for tool_call in example["tool_calls"]:
        openai_tool_calls.append(
            {
                "id": str(uuid.uuid4()),
                "type": "function",
                "function": {
                    # The name of the function right now corresponds
                    # to the name of the pydantic model
                    # This is implicit in the API right now,
                    # and will be improved over time.
                    "name": tool_call.__class__.__name__,
                    "arguments": tool_call.json(),
                },
            }
        )
    messages.append(
        AIMessage(content="", additional_kwargs={"tool_calls": openai_tool_calls})
    )
    tool_outputs = ["You have correctly called this tool."] * len(openai_tool_calls)
    for output, tool_call in zip(tool_outputs, openai_tool_calls):
        messages.append(ToolMessage(content=output, tool_call_id=tool_call["id"]))
    return messages


examples = [
    (
        """2. Demographics

We’ll begin with some questions so that we can get to know you and your family.

[BIRTHYR] What year were you born?

[GENDER] Which gender do you identify with?

Female

Male

Non-binary

Prefer not to answer

[ZIPCODE] What is your zip code?""",
        [
            Question(
                module="2. Demographics",
                question_id="BIRTHYR",
                question="What year were you born?",
                instructions=None,
                options=None,
                language="English"
            ),
            Question(
                module="2. Demographics",
                question_id="GENDER",
                question="Which gender do you identify with?",
                instructions=None,
                options="Female ||| Male ||| Non-binary ||| Prefer not to answer",
                language="English"
            ),
            Question(
                module="2. Demographics",
                question_id="ZIPCODE",
                question="What is your zip code?",
                instructions=None,
                options=None,
                language="English"
            ),
        ]
    ),
    (
        """[EDUCATION] What is the highest level of education you have completed?

o Less than high school

o High school / GED

o Some college

o 2-year college degree

o 4-year college degree

o Vocational training

o Graduate degree

o Prefer not to answer""",
        [
            Question(
                module=None,
                question_id="EDUCATION",
                question="What is the highest level of education you have completed?",
                instructions=None,
                options="Less than high school ||| High school / GED ||| Some college ||| 2-year college degree ||| "
                        "4-year college degree ||| Vocational training ||| Graduate degree ||| Prefer not to answer",
                language="English"
            ),
        ]
    ),
    (
        """And how much do you disagree or agree with the following statements? For each statement, please """
        """rate how much the pair of traits applies to you, even if one trait applies more strongly than the """
        """other. I see myself as... [For each: 1=Strongly disagree, 7=Strongly agree]

[BIG5Q1] Extraverted, enthusiastic

[BIG5Q2] Critical, quarrelsome

[BIG5Q3] Dependable, self-disciplined""",
        [
            Question(
                module=None,
                question_id="BIG5Q1",
                question="And how much do you disagree or agree with the following statement? I see myself "
                         "as... Extraverted, enthusiastic",
                instructions="Please rate how much the pair of traits applies to you, even if one trait applies "
                             "more strongly than the other.",
                options="Strongly disagree ### 1 ||| 2 ||| 3 ||| 4 ||| 5 ||| 6 ||| Strongly agree ### 7",
                language="English"
            ),
            Question(
                module=None,
                question_id="BIG5Q2",
                question="And how much do you disagree or agree with the following statement? I see myself "
                         "as... Critical, quarrelsome",
                instructions="Please rate how much the pair of traits applies to you, even if one trait applies "
                             "more strongly than the other.",
                options="Strongly disagree ### 1 ||| 2 ||| 3 ||| 4 ||| 5 ||| 6 ||| Strongly agree ### 7",
                language="English"
            ),
        ]
    ),
    (
        """4. Savings Habits

Next, we will ask questions over your monthly saving habits and the potential methods that are used to save money.

1. On average, how much money do you spend monthly on essential goods below that contribute to your wellbeing """
        """(explain/ add in an example)

2. How do you typically spend your monthly income? (choose all that may apply)

a. Home and Housing

b. Retirement

c. Bills and Utility

d. Medical (Physical and Mental Treatment and Care)

e. Taxes

f. Insurance

g. Credit Card Payments (if applicable)

h. Food

i. Shopping and personal items

j. Other

k. I am not able to save money each month

l. Nothing

m. Don’t Know

3. Do you contribute the same amount or more to your savings each month?""",
        [
            Question(
                module="4. Savings Habits",
                question_id="1",
                question="On average, how much money do you spend monthly on essential goods below that "
                         "contribute to your wellbeing (explain/ add in an example)",
                instructions=None,
                options=None,
                language="English"
            ),
            Question(
                module="4. Savings Habits",
                question_id="2",
                question="How do you typically spend your monthly income?",
                instructions=None,
                options="Home and Housing ### a ||| Retirement ### b ||| Bills and Utility ### c ||| Medical "
                        "(Physical and Mental Treatment and Care) ### d ||| Taxes ### e ||| Insurance ### f ||| Credit "
                        "Card Payments (if applicable) ### g ||| Food ### h ||| Shopping and personal items ### i ||| "
                        "Other ### j ||| I am not able to save money each month ### k ||| Nothing ### l ||| Don’t "
                        "Know ### m",
                language="English"
            ),
            Question(
                module="4. Savings Habits",
                question_id="3",
                question="Do you contribute the same amount or more to your savings each month?",
                instructions=None,
                options=None,
                language="English"
            ),
        ]
    ),
    (
        """<h1>Round 1, June 2020 Eng</h1>
<table border="1" class="dataframe">
<tbody>
<tr>
<td>Module</td>
<td>Section</td>
<td>Variable</td>
<td>Question</td>
<td>Response set</td>
</tr>
<tr>
<td></td>
<td>CONS. Introduction and Consent</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>CONS</td>
<td></td>
<td>Good morning/afternoon/evening. My name is ______________________ from Innovations from Poverty Action, a """
        """Mexican research NGO. \n \n We would like to invite you to participate in a survey lasting about """
        """20 minutes about the effects of covid-19 on economic and social conditions in the Mexico City """
        """metropolitan area. If you are eligible for the survey we will compensate you [30 pesos] in """
        """airtime for completing your survey.</td>
<td></td>
</tr>
<tr>
<td></td>
<td>CONS</td>
<td>cons1</td>
<td>Can I give you more information?</td>
<td>Y/N</td>
</tr>
<tr>
<td></td>
<td>CONS</td>
<td></td>
<td>*If cons1=N\n Thank you for your response. We will end the survey now.</td>
<td>[End survey]</td>
</tr>
<tr>
<td>core</td>
<td>END</td>
<td>end4</td>
<td>What is your first name?</td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>DEM</td>
<td>dem1</td>
<td>How old are you?</td>
<td>*Enter age*\n ###</td>
</tr>
<tr>
<td>core</td>
<td>CONS</td>
<td></td>
<td>*If DEM1&lt;18*\n Thank you for your response. We will end the survey now.</td>
<td>[End survey]</td>
</tr>""",
        [
            Question(
                module="CONS. Introduction and Consent",
                question_id=None,
                question="""Good morning/afternoon/evening. My name is ______________________ from """
                         """Innovations from Poverty Action, a """
                         """Mexican research NGO. \n \n We would like to invite you to participate in a survey """
                         """lasting about 20 minutes about the effects of covid-19 on economic and social """
                         """conditions in the Mexico City metropolitan area. If you are eligible for the survey """
                         """we will compensate you [30 pesos] in airtime for completing your survey.""",
                instructions=None,
                options=None,
                language="English"
            ),
            Question(
                module="CONS. Introduction and Consent",
                question_id="cons1",
                question="Can I give you more information?",
                instructions="*If cons1=N\n Thank you for your response. We will end the survey now. "
                             "[End survey]",
                options="Y ||| N",
                language="English"
            ),
            Question(
                module="CONS. Introduction and Consent",
                question_id="end4",
                question="What is your first name?",
                instructions=None,
                options=None,
                language="English"
            ),
            Question(
                module="CONS. Introduction and Consent",
                question_id="dem1",
                question="How old are you?",
                instructions="*Enter age*\n ###\n"
                             "*If DEM1&lt;18*\n Thank you for your response. We will end the survey now. "
                             "[End survey]",
                options=None,
                language="English"
            ),
        ]
    ),
    (
        """<tr>
<td>MEXICO</td>
<td>INC</td>
<td>inc11_mex</td>
<td>*If YES to INC12_mex*\n If schools and daycares remained closed and workplaces re-opened, would anyone in your """
        """household have to stay home and not return to work in order to care for children too young to """
        """stay at home without supervision?</td>
<td>*Read out, select multiple possible*\n Grandparents\n Hired babysitter\n Neighbors\n Mother who normally st\n """
        """Mother who normally works outside the home\n Father who normally works outside the home\n Older """
        """sibling\n DNK</td>
</tr>
<tr>
<td></td>
<td>NET. Social Safety Net</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>core</td>
<td>NET</td>
<td>net1</td>
<td>Do you usually receive a regular transfer from any cash transfer or other in-kind social support program?\n """
        """\n HINT: Social safety net programs include cash transfers and in-kind food transfers (food """
        """stamps and vouchers, food rations, and emergency food distribution). Example includes XXX cash """
        """transfer programme.</td>
<td>Y/N/DNK</td>
</tr>""",
        [
            Question(
                module=None,
                question_id="inc11_mex",
                question="""If schools and daycares remained closed and workplaces re-opened, would """
                         """anyone in your household have to stay home and not return to work in order to care for """
                         """children too young to stay at home without supervision?""",
                instructions="*If YES to INC12_mex*\n*Read out, select multiple possible*",
                options="Grandparents ||| Hired babysitter ||| Neighbors ||| Mother who normally st ||| Mother who "
                        "normally works outside the home ||| Father who normally works outside the home ||| "
                        "Older sibling ||| DNK",
                language="English"
            ),
            Question(
                module="NET. Social Safety Net",
                question_id="net1",
                question="""Do you usually receive a regular transfer from any cash transfer or other """
                         """in-kind social support program?\n \n HINT: Social safety net programs """
                         """include cash transfers and in-kind food transfers (food stamps and vouchers, """
                         """food rations, and emergency food distribution). Example includes XXX cash """
                         """transfer programme.""",
                instructions="*If cons1=N\n Thank you for your response. We will end the survey now. "
                             "[End survey]",
                options="Y ||| N ||| DNK",
                language="English"
            ),
        ]
    ),
    (
        """<tr>
<td></td>
<td>POL. POLICING</td>
<td></td>
<td></td>
<td></td>
</tr>
<tr>
<td>MEXICO</td>
<td>POL</td>
<td></td>
<td>Now I am going to ask you some questions about the main problems of insecurity in Mexico City and the """
        """performance of the city police since the coronavirus pandemic began around March 20, 2020.</td>
<td></td>
</tr>
<tr>
<td>MEXICO</td>
<td>POL</td>
<td>POL1</td>
<td>Compared to the level of insecurity that existed in your neighborhood before the pandemic began, do you """
        """consider that the level of insecurity in your neighborhood decreased, remained more or less the """
        """same, or increased?</td>
<td>Decreased\n it was more or less the same\n increased\n (777) Doesn’t answer\n (888) Doesn’t know\n (999) """
        """Doesn’t apply</td>
</tr>""",
        [
            Question(
                module="POL. POLICING",
                question_id="POL1",
                question="""Compared to the level of insecurity that existed in your neighborhood before """
                         """the pandemic began, do you consider that the level of insecurity in your """
                         """neighborhood decreased, remained more or less the same, or increased?""",
                instructions=None,
                options="Decreased ||| it was more or less the same ||| increased ||| Doesn’t answer ### 777 ||| "
                        "Doesn’t know ### 888 ||| Doesn’t apply ### 999",
                language="English"
            ),
        ]
    ),
    (
        """Module: HISTORY1 | Question: 1408 | Language: AFRIKAANS

Wanneer was die laaste keer dat jy ’n Papsmeer gehad het?

WITHIN THE LAST 3 YEARS = 1

4-5 YEARS AGO = 2

6-10 YEARS AGO = 3

MORE THAN 10 YEARS AGO = 4

DON'T KNOW/DON'T REMEMBER = 8""",
        [
            Question(
                module="HISTORY1",
                question_id="1408",
                question="Wanneer was die laaste keer dat jy ’n Papsmeer gehad het?",
                instructions=None,
                options="WITHIN THE LAST 3 YEARS ### 1 ||| 4-5 YEARS AGO ### 2 ||| 6-10 YEARS AGO ### 3 ||| MORE "
                        "THAN 10 YEARS AGO ### 4 ||| DON'T KNOW/DON'T REMEMBER ### 8",
                language="AFRIKAANS"
            ),
        ]
    ),
]


def set_langchain_splits(chunk_size: int, overlap_size: int):
    """
    Set the chunk size and overlap size for the (default) langchain splitter.

    :param chunk_size: Size of each chunk.
    :type chunk_size: int
    :param overlap_size: Size of the overlap.
    :type overlap_size: int
    """

    global langchain_splitter_chunk_size, langchain_splitter_overlap_size
    langchain_splitter_chunk_size = chunk_size
    langchain_splitter_overlap_size = overlap_size


def clean_whitespace(s: str) -> str:
    """
    Clean and standardize whitespace in a string.

    This includes converting tabs to spaces, trimming leading/trailing whitespace from each line,
    reducing multiple spaces to single ones, and reducing more than three consecutive linebreaks to three.

    :param s: String to clean.
    :type s: str
    :return: Cleaned string.
    :rtype: str
    """

    # convert tabs to spaces
    s = re.sub(r'\t', ' ', s)

    # trim leading and trailing whitespace from each line
    lines = s.split('\n')
    trimmed_lines = [line.strip() for line in lines]
    s = '\n'.join(trimmed_lines)

    # reduce multiple spaces to a single space
    s = re.sub(' +', ' ', s)

    # reduce more than two consecutive linebreaks to two
    s = re.sub(r'\n{3,}', '\n\n', s)

    return s


def split_langchain(content, create_doc: bool = True) -> list:
    """
    Split content into chunks using langchain.

    :param content: Content to split.
    :param create_doc: Flag to indicate whether to create a langchain Document.
    :type create_doc: bool
    :return: List of langchain documents.
    :rtype: list
    """

    # split content into chunks, with overlap to ensure that we capture entire questions (at risk of duplication)
    doc = SchemaDocument(page_content=content) if create_doc else content
    return RecursiveCharacterTextSplitter(chunk_size=langchain_splitter_chunk_size,
                                          chunk_overlap=langchain_splitter_overlap_size).split_documents([doc])


def split_nltk(content) -> list:
    """
    Split content into sentences using NLTK.

    :param content: Content to split.
    :return: List of sentences.
    :rtype: list
    """

    return nltk.sent_tokenize(content)


def split_spacy(content) -> list:
    """
    Split content into sentences using spaCy.

    :param content: Content to split.
    :return: List of sentences.
    :rtype: list
    """

    doc = nlp(content)
    return [sent.text for sent in doc.sents]


def split(content, splitter: Callable = split_langchain, create_doc: bool = True) -> list:
    """
    Split content using a specified splitting function.

    :param content: Content to split.
    :param splitter: Function to use for splitting. Defaults to split_langchain.
    :type splitter: Callable
    :param create_doc: Flag to indicate whether to create a langchain Document for splitting.
    :type create_doc: bool
    :return: List of split content.
    :rtype: list
    """

    split_docs = splitter(content, create_doc)
    return [doc.page_content if not isinstance(doc, str) else doc for doc in split_docs]


def read_docx(file_path: str, splitter: Callable = split_langchain) -> list:
    """
    Parse a DOCX file into a list of content chunks using a specified splitting function.

    :param file_path: Path to the DOCX file.
    :type file_path: str
    :param splitter: Function to use for splitting the content. Defaults to split_langchain.
    :type splitter: Callable
    :return: List of split content chunks.
    :rtype: list
    """

    # use langchain/unstructured to parse the DOCX file
    loader = UnstructuredFileLoader(file_path, mode="elements")
    data = loader.load()

    # concatenate page contents, using double-linebreaks as page/element separators
    content = '\n\n'.join([page.page_content for page in data])
    content = clean_whitespace(content)

    return split(content, splitter)


def parse_csv(input_csv_file: str, splitter: Callable = split_langchain) -> dict | list:
    """
    Parse a CSV file into a dictionary with questionnaire data.

    The function handles REDCap data dictionaries explicitly, then falls back to generic handling of other CSV files.

    :param input_csv_file: Path to the CSV file.
    :type input_csv_file: str
    :param splitter: Function to use for splitting in case not a REDCap data dictionary. Defaults to split_langchain.
    :type splitter: Callable
    :return: Dictionary with form data or split content.
    :rtype: dict | list
    """

    form_data = {"questionnairedata": []}
    with open(input_csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        try:
            if 'Field Type' not in reader.fieldnames:
                # raise exception to fall back to generic CSV processing
                raise ValueError('CSV file does not appear to be a REDCap data dictionary (no "Field Type" column).')

            # process REDCap data dictionary
            for row in reader:
                question_data = {}
                field_type = row['Field Type']
                if field_type == 'descriptive' or field_type in ['text', 'number']:
                    question_data['question_id'] = row['Variable / Field Name']
                    question_data['question'] = row['Field Label']
                    question_data['instructions'] = row.get('Field Note', '')
                    question_data['options'] = []
                elif field_type in ['radio', 'checkbox']:
                    question_data['question_id'] = row['Variable / Field Name']
                    question_data['question'] = row['Field Label']
                    question_data['instructions'] = row.get('Field Note', '')
                    choices = row.get('Choices, Calculations, OR Slider Labels', '').split('|')
                    question_data['options'] = ' ||| '.join(
                        [f"{c.split(',')[1].strip()} ### {c.split(',')[0].strip()}" for c in choices])
                else:
                    continue
                form_data['question'].append(question_data)
        except Exception:
            # fall back to generic CSV processing
            content_list = ['\t'.join(row) for row in reader]
            content_str = '\n'.join(content_list)
            content_str = clean_whitespace(content_str)
            return split(content_str, splitter)

    return form_data


def convert_tables(md_text: str) -> str:
    """
    Convert Markdown tables to plain text format.

    :param md_text: Markdown text containing tables.
    :type md_text: str
    :return: Plain text representation of tables.
    :rtype: str
    """

    table_blocks = re.findall(r'\|[\s\S]+?\|', md_text)
    plain_text_tables = ""
    for table_block in table_blocks:
        rows = table_block.strip().split('\n')
        plain_text_tables += '\n'.join(
            ['\t'.join(cell.strip() for cell in row.split('|')[1:-1]) for row in rows]) + '\n\n'

    return plain_text_tables


def process_in_korn_format(data: list) -> dict:
    """
    Process data in a specific format to structure it into a dictionary with questions.

    :param data: List of modules and questions.
    :type data: list
    :return: Dictionary with structured questions.
    :rtype: dict
    """

    final_data = {'questionnairedata': []}
    for module in data:
        for question in module['questions']:
            final_data['questionnairedata'].append({
                'module': module.get('moduleTitle', ''),
                'question': question['question'],
                'instructions': question['instructions'],
                'options': question['options']
            })
    return final_data


def parse_xlsx(file_path: str, splitter: Callable = split_langchain):
    """
    Parse an XLSX file into a dictionary with questionnaire data.

    This function starts by assuming that the XLSX file is in XLSForm format, then falls back to generic parsing
    in the case where XLSForm parsing fails.

    :param file_path: Path to the XLSX file.
    :type file_path: str
    :param splitter: Function used to split content. Defaults to split_langchain.
    :type splitter: Callable
    :return: List of processed content in Document format or list format.
    """

    try:
        # define input tags for XML parsing
        input_tags = [
            "{http://www.w3.org/2002/xforms}input",
            "{http://www.w3.org/2002/xforms}select1",
            "{http://www.w3.org/2002/xforms}textarea",
            "{http://www.w3.org/2002/xforms}upload"
        ]

        # convert XLSX to XML (assuming XLSForm format)
        path_without_ext = os.path.splitext(file_path)[0]
        os.system(f'xls2xform "{file_path}" "{path_without_ext}.xml"')
        tree = ElementTree.parse(f"{path_without_ext}.xml")

        # initialize questionnaire processing
        questionnaire = []
        start = False
        state = ""

        # iterate over XML tree elements
        for elem in tree.iter():
            if elem.tag == "{http://www.w3.org/1999/xhtml}body":
                start = True
            if start:
                if elem.tag == "{http://www.w3.org/2002/xforms}group":
                    questionnaire.append({"module": "", "questions": []})
                    state = "group"
                if elem.tag in input_tags and not questionnaire:
                    questionnaire.append({"module": "", "questions": []})
                if elem.tag in ["{http://www.w3.org/2002/xforms}input", "{http://www.w3.org/2002/xforms}textarea",
                                "{http://www.w3.org/2002/xforms}upload"]:
                    state = "input"
                    questionnaire[-1]["questions"].append({"question": "", "instructions": "", "options": []})
                if elem.tag == "{http://www.w3.org/2002/xforms}select1":
                    state = "select"
                    questionnaire[-1]["questions"].append({"question": "", "instructions": "", "options": []})
                if elem.tag == "{http://www.w3.org/2002/xforms}item" and state == "select":
                    state = "option"
                    questionnaire[-1]["questions"][-1]["options"].append({"value": "", "label": ""})
                if elem.tag == "{http://www.w3.org/2002/xforms}label":
                    if state == "group":
                        questionnaire[-1]["module"] = elem.text
                    elif state in ["input", "select"]:
                        questionnaire[-1]["questions"][-1]["question"] = elem.text
                    elif state == "option":
                        questionnaire[-1]["questions"][-1]["options"][-1]["label"] = elem.text
                if elem.tag == "{http://www.w3.org/2002/xforms}value" and state == "option":
                    questionnaire[-1]["questions"][-1]["options"][-1]["value"] = elem.text
                    state = "select"
        return process_in_korn_format(questionnaire)
    except Exception:
        # fallback method for processing XLSX files (when XLSForm processing fails)
        loader = UnstructuredExcelLoader(file_path, mode="elements")
        data = loader.load()
        content_list = []
        for page in data:
            if 'text_as_html' in page.metadata:
                html_content = page.metadata['text_as_html']
            else:
                html_content = page.page_content
            if 'page_name' in page.metadata:
                page_name = page.metadata['page_name']
                content_list.append('<h1>' + page_name + '</h1>\n' + html_content)

        # split and return the processed content
        split_content = []
        for content in content_list:
            split_content.extend(split(clean_whitespace(content), splitter))
        return split_content


def read_local_html(path: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process local HTML file into a structured format.

    This function reads an HTML file from a local path, converts it to markdown,
    and then splits it into a structured format using a specified splitter function.

    :param path: Path to the local HTML file.
    :type path: str
    :param splitter: Function to split the processed markdown into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    """

    with open(path, 'r', encoding='utf-8') as file:
        doc = SchemaDocument(page_content=file.read())
        md = MarkdownifyHTMLProcessor().process(doc)
        return splitter(md, False)


def read_html(url: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process HTML content from a URL into a structured format.

    This function fetches HTML content from a given URL, converts it to markdown,
    and then splits it into a structured format using a specified splitter function.

    :param url: URL of the HTML page to be read.
    :type url: str
    :param splitter: Function to split the processed markdown into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    :rtype: list
    """

    response = requests.get(url)
    doc = SchemaDocument(page_content=response.text)
    md = MarkdownifyHTMLProcessor().process(doc)
    return splitter(md, False)


def read_pdf_pypdf(path: str, splitter: Callable = split_langchain, split_content: bool = True) -> list:
    """
    Read and process PDF content into a structured format using PyPDF.

    This function reads a PDF file, extracts text from each page, and optionally
    splits it into a structured format using a specified splitter function.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param split_content: Boolean flag to determine whether to split the content or not.
    :type split_content: bool
    :return: List of processed (and optionally split) content.
    :rtype: list
    """

    content = []
    reader = PdfReader(path)
    for idx, page in enumerate(reader.pages):
        text = page.extract_text()
        content.append(text)
    return splitter('\n\n'.join(content)) if split_content else content


def extract_text_from_pdf_tabula(file_path: str, splitter: Callable = split_langchain,
                                 split_content: bool = True) -> list:
    """
    Extract and process text from a PDF file using Tabula for table extraction.

    This function reads a PDF file and extracts text from tables on each page using Tabula. It then cleans the
    extracted tables by dropping empty columns, filling NaN values, and combining them into a text format.
    Optionally, the extracted text can be split into a structured format using a specified splitter function.

    :param file_path: Path to the PDF file.
    :type file_path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param split_content: Boolean flag to determine whether to split the content or not.
    :type split_content: bool
    :return: List of processed (and optionally split) content from the PDF tables.
    :rtype: list
    """

    with open(file_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages)

    content = []
    for page in range(1, num_pages + 1):
        # read PDF tables with Tabula
        df = read_pdf(file_path, pages=page, multiple_tables=True)

        # clean and process tables
        cleaned_tables = clean_tables(df)

        # convert cleaned tables to text
        text = convert_tables_to_text(cleaned_tables)
        content.append(text)

    return splitter('\n\n'.join(content)) if split_content else content


def clean_tables(dataframes: list) -> list:
    """
    Clean and process extracted tables from a PDF.

    :param dataframes: List of DataFrame objects representing extracted tables.
    :type dataframes: list
    :return: List of cleaned and processed DataFrame objects.
    :rtype: list
    """

    cleaned_tables = []
    for table in dataframes:
        # drop columns with all NaN values and replace NaN with empty string
        table.dropna(axis=1, how='all', inplace=True)
        table.fillna('', inplace=True)
        # append non-empty tables to the list
        if not table.empty:
            cleaned_tables.append(table)
    return cleaned_tables


def convert_tables_to_text(tables: list) -> str:
    """
    Convert cleaned tables to a text format.

    :param tables: List of cleaned DataFrame objects representing tables.
    :type tables: list
    :return: String representation of the tables.
    :rtype: str
    """

    text = ''
    for table in tables:
        # convert DataFrame to string and replace multiple spaces
        table_string = table.to_string(index=False, header=True)
        table_string = re.sub(' +', ' ', table_string)
        text += table_string + '\n\n\n'
    return text


def read_ocr(path: str, splitter: Callable = split_langchain) -> list:
    """
    Read and process text from an image-based PDF file using OCR.

    This function converts each page of a PDF to an image and then uses OCR to extract text.
    The extracted text is then split into a structured format using a specified splitter function.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :return: List of processed and split content.
    :rtype: list
    """

    pages = convert_from_path(path)
    content = ''
    for page in pages:
        text = pytesseract.image_to_string(page)
        content += text
    return splitter(content)


def read_pdf_combined(path: str, splitter: Callable = split_langchain, min_length: int = 600) -> list:
    """
    Read and process text from a PDF file combining PyPDF and Tabula extraction methods.

    This function extracts text from a PDF using both PyPDF and Tabula methods. The combined
    text is then checked for a minimum length. If the length requirement is met, the text is
    split into a structured format using a specified splitter function. If not, it falls back
    to OCR.

    :param path: Path to the PDF file.
    :type path: str
    :param splitter: Function to split the extracted text into a structured format.
    :type splitter: Callable
    :param min_length: Minimum expected length for the combined text.
    :type min_length: int
    :return: List of processed and split content, or content from OCR if length requirement is not met.
    :rtype: list
    """

    text_pypdf_per_page = read_pdf_pypdf(path, split_langchain, False)
    text_tabula_per_page = extract_text_from_pdf_tabula(path, split_langchain, False)

    combined_text = combine_texts(text_pypdf_per_page, text_tabula_per_page)

    # clean whitespace and check for minimum length
    combined_text = clean_whitespace(combined_text)
    if len(combined_text.strip()) >= min_length:
        return splitter(combined_text)

    return read_ocr(path, splitter)


def combine_texts(texts1: list, texts2: list) -> str:
    """
    Combine texts from two lists, appending text from the second list to the first.

    :param texts1: List of strings from the first source.
    :type texts1: list
    :param texts2: List of strings from the second source.
    :type texts2: list
    :return: Combined text as a single string.
    :rtype: str
    """

    combined_text = ""
    for text1, text2 in zip(texts1, texts2):
        combined_text += text1 + "\n" + text2 + "\n"
    return combined_text


def generate_extractor_chain(model_input: str, api_base: str, openai_api_key: str, open_api_version: str,
                             provider: str = "azure") -> Runnable:
    """
    Generate an extractor chain based on the specified language model and API settings.

    :param model_input: Name of the language model to use.
    :type model_input: str
    :param api_base: Base URL for the API.
    :type api_base: str
    :param openai_api_key: API key for accessing OpenAI services.
    :type openai_api_key: str
    :param open_api_version: Version of the OpenAI API to use.
    :type open_api_version: str
    :param provider: Provider for the LLM service ("openai" for direct OpenAI, "azure" for Azure). Default is "azure".
    :type provider: str
    :return: An extraction chain configured with the specified parameters.
    :rtype: Runnable
    """

    # initialize LLM
    if provider == "azure":
        llm = AzureChatOpenAI(
            temperature=0,
            verbose=True,
            model_name=model_input,
            azure_endpoint=api_base,
            deployment_name=model_input,
            openai_api_version=open_api_version,
            openai_api_key=openai_api_key,
            openai_api_type="azure",
        )
    elif provider == "openai":
        llm = ChatOpenAI(
            temperature=0,
            verbose=True,
            model_name=model_input,
            openai_api_key=openai_api_key
        )
    else:
        raise ValueError("Unsupported provider specified. Choose 'openai' or 'azure'.")

    runnable = prompt | llm.with_structured_output(
        schema=QuestionList,
        method="function_calling",
        include_raw=False,
    )

    return runnable


def uri_validator(url: str) -> bool:
    """
    Validate if the given string is a valid URI.

    :param url: The string to validate as URI.
    :type url: str
    :return: True if the string is a valid URI, False otherwise.
    :rtype: bool
    """

    result = urlparse(url)
    return all([result.scheme, result.netloc])


def get_data_from_url(url: str, splitter: Callable = split_langchain) -> list[str] | dict:
    """
    Fetch and process data from a given URL based on file extension.

    :param url: URL of the file to process.
    :type url: str
    :param splitter: Function or method used for processing the data.
    :type splitter: Callable
    :return: Processed data based on file extension or None if an error occurs.
    :rtype: list[str] | dict | None
    """

    valid_url = uri_validator(url)
    response = []
    if valid_url:
        response = requests.get(url.strip())
        if response.status_code != 200:
            parser_logger.log(logging.ERROR, f'Error: {response.status_code}')
            return []

    extension = url.split('.')[-1].strip().lower()
    process_functions = {
        'pdf': read_pdf_combined,
        'docx': read_docx,
        'csv': parse_csv,
        'xlsx': parse_xlsx,
        'html': read_local_html,
        'txt': read_docx
    }

    if extension in process_functions:
        # process based on the extension
        process_function = process_functions.get(extension, read_html)
        if valid_url:
            # if the URL is valid, save the file locally
            temp_file_handle, temp_file_path = tempfile.mkstemp(suffix='.' + extension)
            with os.fdopen(temp_file_handle, 'wb') as temp_file:
                temp_file.write(response.content)
            file_to_process = temp_file_path
        else:
            file_to_process = url
        retval = process_function(file_to_process, splitter)
    else:
        # for other extensions, assume it's a URL to an HTML page
        retval = read_html(url, splitter)

    # if the result is a list of documents, extract the page content for a list of strings
    if isinstance(retval, list) and retval:
        retval = [doc.page_content if not isinstance(doc, str) else doc for doc in retval]
    return retval


def total_string_length(d: dict) -> int:
    """
    Calculate the total string length of all values in a dictionary.

    :param d: The dictionary whose values' string lengths are to be summed.
    :type d: dict
    :return: The total string length of all values.
    :rtype: int
    """

    # sum length of all strings, including those in nested lists of dicts
    total_length = 0
    for value in d.values():
        if isinstance(value, list):
            for item in value:
                if isinstance(item, dict):
                    total_length += total_string_length(item)
        else:
            total_length += len(str(value))

    return total_length


def clean_data(data: dict) -> dict:
    """
    Clean the provided data by removing empty modules and questions — and by removing duplicate questions,
    keeping the ones with more content.

    :param data: The data to be cleaned.
    :type data: dict
    :return: The cleaned data.
    :rtype: dict
    """

    cleaned_data = {}
    for key, dt in data.items():
        # assemble module questions, dropping empty questions
        module = {question: question_data for question, question_data in dt.items() if question and question_data}

        # skip empty modules
        if not module:
            parser_logger.log(logging.INFO, f"Skipping empty module: {key}")
        else:
            parser_logger.log(logging.INFO, f"* Module: {key}")
            # for each question, look for questions with the same language
            for question, question_data in module.items():
                if question_data:
                    to_remove = set()
                    for i in range(len(question_data)):
                        for j in range(i + 1, len(question_data)):
                            if question_data[i]['language'] == question_data[j]['language']:
                                # compare total string lengths and mark the shorter one for removal
                                length_i = total_string_length(question_data[i])
                                length_j = total_string_length(question_data[j])
                                if length_i > length_j:
                                    to_remove.add(j)
                                else:
                                    to_remove.add(i)
                                parser_logger.log(logging.INFO, f"  * Dropping duplicate question with shorter "
                                                                f"content: "
                                                                f"{question} - {question_data[i]['question']} "
                                                                f"({length_i} vs {length_j})")

                    # remove duplicates after identifying them
                    for index in sorted(to_remove, reverse=True):
                        question_data.pop(index)

            cleaned_data[key] = module

    return cleaned_data


async def extract_data(chain: Runnable, url: str, replacement_examples: List[Tuple[str, List[Question]]] = None,
                       additional_examples: List[Tuple[str, List[Question]]] = None) -> dict:
    """
    Asynchronously process the content from a given URL and parse it into structured data.

    :param chain: The AI prediction chain to be used.
    :type chain: Runnable
    :param url: URL of the document to process.
    :type url: str
    :param replacement_examples: Replacement examples to use (if any).
    :type replacement_examples: List[Tuple[str, List[Question]]]
    :param additional_examples: Additional examples to use (if any).
    :type additional_examples: List[Tuple[str, List[Question]]]
    :return: Dictionary of questions, organized by module.
    :rtype: dict
    """

    # construct example messages, including default set, plus any replacements or additional examples
    example_messages = []
    if replacement_examples is not None:
        for text, tool_call in replacement_examples:
            example_messages.extend(
                tool_example_to_messages({"input": text, "tool_calls": tool_call})
            )
    else:
        for text, tool_call in examples:
            example_messages.extend(
                tool_example_to_messages({"input": text, "tool_calls": tool_call})
            )
    if additional_examples is not None:
        for text, tool_call in additional_examples:
            example_messages.extend(
                tool_example_to_messages({"input": text, "tool_calls": tool_call})
            )

    # get data from the URL
    docs = get_data_from_url(url)

    # process data
    results = []
    if isinstance(docs, list):
        # process list of documents asynchronously
        if docs:
            # track our LLM usage with an OpenAI callback
            with get_openai_callback() as cb:
                # create a list of tasks, then execute them asynchronously
                tasks = [chain.ainvoke({"text": page, "examples": example_messages}) for page in docs]
                results = await asyncio.gather(*tasks)

                # report LLM usage
                parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
                parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
                parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
                parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
                parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")
    else:
        # process single document, tracking our LLM usage with an OpenAI callback
        with get_openai_callback() as cb:
            results = [chain.ainvoke({"text": docs, "examples": example_messages})]

            # report LLM usage
            parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
            parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
            parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
            parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
            parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")

    # organize questions by module and question ID
    grouped_content = {}
    question_module = {}
    current_module = '(none)'
    unknown_id_count = 0
    for res in results:
        for question in res.questions:
            # get module name, defaulting to the current one if none specified
            module = question.module if question.module else current_module

            # process question, if any
            if question.question and question.question.strip():
                # get question ID, if available
                question_id = question.question_id.strip() if question.question_id else ''
                if not question_id:
                    # if no question ID is provided, generate a unique ID
                    unknown_id_count += 1
                    question_id = f"unknown_id_{unknown_id_count}"

                # always keep questions with the same ID together in the same module
                if question_id in question_module:
                    module = question_module[question_id]
                else:
                    question_module[question_id] = module

                # parse and organize options, if any
                options = []
                if question.options:
                    option_strs = [option.strip() for option in question.options.split('|||')]
                    for option_str in option_strs:
                        if '###' in option_str:
                            option_parts = option_str.split('###')
                            if len(option_parts) == 2:
                                option_parts = [part.strip() for part in option_parts]
                                options.append({'label': option_parts[0], 'value': option_parts[1]})
                            else:
                                options.append({'label': option_str, 'value': option_str})
                        else:
                            options.append({'label': option_str, 'value': option_str})

                # add question, grouped by module and question ID
                grouped_content.setdefault(module, {}).setdefault(question_id, [])
                grouped_content[module][question_id].append({
                    'question': question.question,
                    'language': question.language if question.language else '',
                    'options': options,
                    'instructions': question.instructions if question.instructions else '',
                })

            # remember current module for next question
            current_module = module

    # return cleaned-up version of the data
    return clean_data(grouped_content)


async def extract_data_from_directory(path_to_ingest: str, chain: LLMChain) -> list:
    """
    Extract structured data from all files in a specified directory.

    :param path_to_ingest: Path to the directory containing files to process.
    :type path_to_ingest: str
    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :return: A list of structured data from all processed files.
    :rtype: list
    """

    data_list = []
    for root, dirs, files in os.walk(path_to_ingest):
        for file in files:
            if file.endswith((".DS_Store", ".db")):
                # skip system files
                continue

            file_path = os.path.join(root, file)
            data_list.append(await extract_data(chain, file_path))

    return data_list
