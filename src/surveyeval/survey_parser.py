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

from surveyeval.html_tools import MarkdownifyHTMLProcessor
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
import pytesseract
from pypdf import PdfReader
from tabula.io import read_pdf
from pdf2image import convert_from_path
from langchain.schema import Document as SchemaDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader
import importlib.resources as pkg_resources
from openpyxl import load_workbook

# initialize global resources
parser_logger = logging.getLogger(__name__)
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')
empty_form_path = str(pkg_resources.files('surveyeval').joinpath('resources/EmptyForm.xlsx'))

# initialize global variables
langchain_splitter_chunk_size = 3000
langchain_splitter_overlap_size = 500

# initialize parsing schema
module_name_spec = ('The short name or identifier of the survey module, if any, which is the main section, category, '
                    'or group within which a questionnaire or digital form\'s questions or form fields are located. '
                    'Examples of module names: "demographics", "health", "hhmembers", "factors", etc.')
module_title_spec = ('The longer title of the survey module, if any. Examples of module titles: "Household '
                     'demographics", "People living in the household", "Factors that influence take-up", etc.')
module_intro_spec = ('The introductory text or instructions that appear at the start of the module, before the '
                     'module\'s first question begins. For example: "Now we\'re going to ask some questions about '
                     'the members of your household."')
questions_spec = ('The list of questions or form fields within the module, including the question ID, question text, '
                  'instructions, response options, and language.')
question_id_spec = ('The numeric or alphanumeric identifier or short variable name identifying the '
                    'question, usually located just before or at the beginning of the question.')
question_spec = ('The exact text of the question or form field, including any introductory text that provides '
                 'context or explanation. Often follows a unique question ID of some sort, like "2.01." or "gender:". '
                 'Should not include response options, which should be included in the "options" field, or extra '
                 'enumerator or interviewer instructions (including interview probes), which should be included in the '
                 '"instructions" field. Be careful: the same question might be asked in multiple languages, and each '
                 'translation should be included as a separate question with the proper language name in the '
                 '"language" field. Never translate between languages or otherwise alter the question text in any way.')
instructions_spec = ('Instructions or other guidance about how to ask or answer the '
                     'question, including enumerator or interviewer instructions. If the question includes '
                     'a list of specific response options, do NOT include those in the instructions. However, if '
                     'there is guidance as to how to fill out an open-ended numeric or text response, or guidance '
                     'about how to choose among the options, include that guidance here.')
options_spec = ("The list of specific response options for multiple-choice questions, "
                "including both the label and the internal value (if specified) for each option. For example, "
                "a 'Male' label might be coupled with an internal value of '1', 'M', or even 'male'. "
                "Separate response options with a space, three pipe symbols ('|||'), and another space, and, "
                "if there is an internal value, add a space, three # symbols ('###'), and the internal value "
                "at the end of the label. For example: 'Male ### 1 ||| Female ### 2' (codes included) or "
                "'Male ||| Female' (no codes); 'Yes ### yes ||| No ### no', 'Yes ### 1 ||| No ### 0', 'Yes ### "
                "y ||| No ### n', or 'YES ||| NO'. Do NOT include fill-in-the-blank content here, only "
                "multiple-choice options. If the question is open-ended, leave this field blank. ")
language_spec = 'The primary language in which the question text is written.'


class Question(BaseModel):
    """Information about a survey question or form field."""

    question_id: Optional[str] = Field(..., description=question_id_spec)
    question: Optional[str] = Field(..., description=question_spec)
    instructions: Optional[str] = Field(..., description=instructions_spec)
    options: Optional[str] = Field(..., description=options_spec)
    language: Optional[str] = Field(..., description=language_spec)


class Module(BaseModel):
    """Information about a survey module, which is the main section, category, or group within which a questionnaire
or digital form's questions or form fields are located. Examples of modules include: "Demographics", "Health",
"Household members", "Education", etc. All questions are located within modules, even if there is no name, title, or
introductory text for the module."""

    module_name: Optional[str] = Field(..., description=module_name_spec)
    module_title: Optional[str] = Field(..., description=module_title_spec)
    module_intro: Optional[str] = Field(..., description=module_intro_spec)
    questions: Optional[List[Question]] = Field(..., description=questions_spec)


class ModuleList(BaseModel):
    """List of extracted survey modules, including all extracted questions or form fields, from a questionnaire or
digital form."""

    modules: List[Module]


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
            Module(
                module_name=None,
                module_title="2. Demographics",
                module_intro="We’ll begin with some questions so that we can get to know you and your family.",
                questions=[
                    Question(
                        question_id="BIRTHYR",
                        question="What year were you born?",
                        instructions=None,
                        options=None,
                        language="English"
                    ),
                    Question(
                        question_id="GENDER",
                        question="Which gender do you identify with?",
                        instructions=None,
                        options="Female ||| Male ||| Non-binary ||| Prefer not to answer",
                        language="English"
                    ),
                    Question(
                        question_id="ZIPCODE",
                        question="What is your zip code?",
                        instructions=None,
                        options=None,
                        language="English"
                    ),
                ]
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
            Module(
                module_name=None,
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="EDUCATION",
                        question="What is the highest level of education you have completed?",
                        instructions=None,
                        options="Less than high school ||| High school / GED ||| Some college ||| 2-year college "
                                "degree ||| 4-year college degree ||| Vocational training ||| Graduate degree ||| "
                                "Prefer not to answer",
                        language="English"
                    ),
                ]
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
            Module(
                module_name=None,
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="BIG5Q1",
                        question="And how much do you disagree or agree with the following statement? I see myself "
                                 "as... Extraverted, enthusiastic",
                        instructions="Please rate how much the pair of traits applies to you, even if one trait "
                                     "applies more strongly than the other.",
                        options="Strongly disagree ### 1 ||| 2 ||| 3 ||| 4 ||| 5 ||| 6 ||| Strongly agree ### 7",
                        language="English"
                    ),
                    Question(
                        question_id="BIG5Q2",
                        question="And how much do you disagree or agree with the following statement? I see myself "
                                 "as... Critical, quarrelsome",
                        instructions="Please rate how much the pair of traits applies to you, even if one trait "
                                     "applies more strongly than the other.",
                        options="Strongly disagree ### 1 ||| 2 ||| 3 ||| 4 ||| 5 ||| 6 ||| Strongly agree ### 7",
                        language="English"
                    ),
                    Question(
                        question_id="BIG5Q3",
                        question="And how much do you disagree or agree with the following statement? I see myself "
                                 "as... Dependable, self-disciplined",
                        instructions="Please rate how much the pair of traits applies to you, even if one trait "
                                     "applies more strongly than the other.",
                        options="Strongly disagree ### 1 ||| 2 ||| 3 ||| 4 ||| 5 ||| 6 ||| Strongly agree ### 7",
                        language="English"
                    ),
                ]
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
            Module(
                module_name=None,
                module_title="4. Savings Habits",
                module_intro="Next, we will ask questions over your monthly saving habits and the potential methods "
                             "that are used to save money.",
                questions=[
                    Question(
                        question_id="1",
                        question="On average, how much money do you spend monthly on essential goods below that "
                                 "contribute to your wellbeing (explain/ add in an example)",
                        instructions=None,
                        options=None,
                        language="English"
                    ),
                    Question(
                        question_id="2",
                        question="How do you typically spend your monthly income?",
                        instructions=None,
                        options="Home and Housing ### a ||| Retirement ### b ||| Bills and Utility ### c ||| Medical "
                                "(Physical and Mental Treatment and Care) ### d ||| Taxes ### e ||| Insurance ### f "
                                "||| Credit Card Payments (if applicable) ### g ||| Food ### h ||| Shopping and "
                                "personal items ### i ||| Other ### j ||| I am not able to save money each month ### "
                                "k ||| Nothing ### l ||| Don’t Know ### m",
                        language="English"
                    ),
                    Question(
                        question_id="3",
                        question="Do you contribute the same amount or more to your savings each month?",
                        instructions=None,
                        options=None,
                        language="English"
                    ),
                ]
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
            Module(
                module_name="CONS",
                module_title="Introduction and Consent",
                module_intro="Good morning/afternoon/evening. My name is ______________________ from "
                             "Innovations from Poverty Action, a "
                             "Mexican research NGO. \n \n We would like to invite you to participate in a survey "
                             "lasting about 20 minutes about the effects of covid-19 on economic and social "
                             "conditions in the Mexico City metropolitan area. If you are eligible for the survey "
                             "we will compensate you [30 pesos] in airtime for completing your survey.",
                questions=[
                    Question(
                        question_id="cons1",
                        question="Can I give you more information?",
                        instructions="*If cons1=N\n Thank you for your response. We will end the survey now. "
                                     "[End survey]",
                        options="Y ||| N",
                        language="English"
                    ),
                    Question(
                        question_id="end4",
                        question="What is your first name?",
                        instructions=None,
                        options=None,
                        language="English"
                    ),
                    Question(
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
            Module(
                module_name=None,
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="inc11_mex",
                        question="If schools and daycares remained closed and workplaces re-opened, would "
                                 "anyone in your household have to stay home and not return to work in order to care "
                                 "for children too young to stay at home without supervision?",
                        instructions="*If YES to INC12_mex*\n*Read out, select multiple possible*",
                        options="Grandparents ||| Hired babysitter ||| Neighbors ||| Mother who normally st ||| Mother "
                                "who normally works outside the home ||| Father who normally works outside the home "
                                "||| Older sibling ||| DNK",
                        language="English"
                    ),
                ]
            ),
            Module(
                module_name="NET",
                module_title="Social Safety Net",
                module_intro=None,
                questions=[
                    Question(
                        question_id="net1",
                        question="Do you usually receive a regular transfer from any cash transfer or other "
                                 "in-kind social support program?\n \n HINT: Social safety net programs "
                                 "include cash transfers and in-kind food transfers (food stamps and vouchers, "
                                 "food rations, and emergency food distribution). Example includes XXX cash "
                                 "transfer programme.",
                        instructions="*If cons1=N\n Thank you for your response. We will end the survey now. "
                                     "[End survey]",
                        options="Y ||| N ||| DNK",
                        language="English"
                    ),
                ]
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
            Module(
                module_name="POL",
                module_title="POLICING",
                module_intro="Now I am going to ask you some questions about the main problems of insecurity in Mexico "
                             "City and the performance of the city police since the coronavirus pandemic began around "
                             "March 20, 2020.",
                questions=[
                    Question(
                        question_id="POL1",
                        question="Compared to the level of insecurity that existed in your neighborhood before "
                                 "the pandemic began, do you consider that the level of insecurity in your "
                                 "neighborhood decreased, remained more or less the same, or increased?",
                        instructions=None,
                        options="Decreased ||| it was more or less the same ||| increased ||| Doesn’t answer ### 777 "
                                "||| Doesn’t know ### 888 ||| Doesn’t apply ### 999",
                        language="English"
                    ),
                ]
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
            Module(
                module_name="HISTORY1",
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="1408",
                        question="Wanneer was die laaste keer dat jy ’n Papsmeer gehad het?",
                        instructions=None,
                        options="WITHIN THE LAST 3 YEARS ### 1 ||| 4-5 YEARS AGO ### 2 ||| 6-10 YEARS AGO ### 3 ||| "
                                "MORE THAN 10 YEARS AGO ### 4 ||| DON'T KNOW/DON'T REMEMBER ### 8",
                        language="AFRIKAANS"
                    ),
                ]
            ),
        ]
    ),
    (
        """Module: FACTORS | Question: 1401a | Language: English

Many different factors can prevent women from getting medical advice or treatment for themselves. When you are """
        """sick and want to get medical advice or treatment, is the following a big problem or not a big problem: """
        """Getting permission to go to the doctor?

BIG PROBLEM = 1

NOT A BIG PROBLEM = 2""",
        [
            Module(
                module_name="FACTORS",
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="1401a",
                        question="Many different factors can prevent women from getting medical advice or treatment "
                                 "for themselves. When you are sick and want to get medical advice or treatment, is "
                                 "the following a big problem or not a big problem: Getting permission to go to the "
                                 "doctor?",
                        instructions=None,
                        options="BIG PROBLEM ### 1 ||| NOT A BIG PROBLEM ### 2",
                        language="English"
                    ),
                ]
            ),
        ]
    ),
    (
        """Module: HISTORY3 | Question: 1454 | Language: AFRIKAANS

Hoeveel keer het dit in die afgelope 12 maande met jou gebeur?

NUMBER OF TIMES: _______""",
        [
            Module(
                module_name="HISTORY3",
                module_title=None,
                module_intro=None,
                questions=[
                    Question(
                        question_id="1454",
                        question="Hoeveel keer het dit in die afgelope 12 maande met jou gebeur?",
                        instructions="NUMBER OF TIMES:",
                        options=None,
                        language="AFRIKAANS"
                    ),
                ]
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
    :return: Dictionary with form data (if REDCap), otherwise list with split content.
    :rtype: dict | list
    """

    with open(input_csv_file, newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)

        # assume REDCap data dictionary format if "Field Type" column is present
        if 'Field Type' in reader.fieldnames:
            # process REDCap data dictionary
            all_questions = {}
            for row in reader:
                assert isinstance(row, dict)  # (type hint for IDE)
                field_type = row['Field Type']
                # only read supported field types
                if field_type in ['descriptive', 'text', 'number', 'radio', 'checkboxes', 'dropdown', 'slider',
                                  'yesno', 'truefalse']:
                    question_id = row['Variable / Field Name']
                    options = []
                    if field_type in ['radio', 'checkboxes', 'dropdown']:
                        # parse choices from "value, label | value, label | ..." pairs
                        choice_strs = row.get('Choices, Calculations, OR Slider Labels', '').split('|')
                        for choice_str in choice_strs:
                            value, label = choice_str.split(',', 1)
                            value = value.strip()
                            label = label.strip()
                            options.append({'label': label, 'value': value})
                    elif field_type == 'slider':
                        # parse choices from "label | label | ..." format
                        slider_strs = row.get('Choices, Calculations, OR Slider Labels', '').split('|')
                        for slider_str in slider_strs:
                            options.append({'label': slider_str, 'value': slider_str})
                    elif field_type == 'yesno':
                        options.append({'label': "yes", 'value': "1"})
                        options.append({'label': "no", 'value': "0"})
                    elif field_type == 'truefalse':
                        options.append({'label': "true", 'value': "1"})
                        options.append({'label': "false", 'value': "0"})

                    all_questions[question_id] = [
                        {
                            'question': row['Field Label'],
                            'instructions': row.get('Field Note', ''),
                            'options': options,
                            'language': "Unknown"   # (language not specified in REDCap data dictionary)
                        }
                    ]

                # return all questions in a single module
                form_data = {
                    "REDCap_module": {
                        "module_name": "REDCap_module",
                        "module_title": "",
                        "module_intro": "",
                        "questions": all_questions
                    }
                }
                return form_data
        else:
            # fall back to generic CSV processing
            content_list = ['\t'.join(row) for row in reader]
            content_str = '\n'.join(content_list)
            content_str = clean_whitespace(content_str)
            return split(content_str, splitter)


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


def parse_xlsx(file_path: str, splitter: Callable = split_langchain) -> list[str] | dict:
    """
    Parse an XLSX file into a dictionary with questionnaire data.

    This function starts by assuming that the XLSX file is in XLSForm format, then falls back to generic parsing
    in the case where XLSForm parsing fails.

    :param file_path: Path to the XLSX file.
    :type file_path: str
    :param splitter: Function used to split content. Defaults to split_langchain.
    :type splitter: Callable
    :return: A dict with structured content if it was an XLSForm, otherwise a list of split content to process.
    :rtype: list[str] | dict
    """

    # first load the workbook
    wb = load_workbook(file_path)

    # if there are survey, choices, and settings worksheets, assume it's an XLSForm
    if 'survey' in wb.sheetnames and 'choices' in wb.sheetnames and 'settings' in wb.sheetnames:
        # process the XLSForm
        survey_ws = wb['survey']
        survey_columns = _get_columns_from_headers(survey_ws)
        survey_data = [row for row in survey_ws.values][1:]
        choices_ws = wb['choices']
        choices_columns = _get_columns_from_headers(choices_ws)
        choices_data = [row for row in choices_ws.values][1:]
        settings_ws = wb['settings']
        settings_columns = _get_columns_from_headers(settings_ws)

        # hack for naming flexibility: if there's a "name" column but no "value" column, rename "name" to "value"
        if 'name' in choices_columns and 'value' not in choices_columns:
            choices_columns['value'] = choices_columns['name']
            del choices_columns['name']

        # read the default_language from row 2 of the settings worksheet
        default_language = settings_ws.cell(row=2, column=settings_columns['default_language']+1).value

        # zip through the survey sheet to create a list of other languages in the form
        other_languages = []
        for column in survey_columns:
            match = re.match(r"^label:(.*)", column)
            if match:
                language = match.group(1).strip()
                # only consider adding languages we haven't already added (no dups wanted)
                if language not in other_languages:
                    # only add languages that also have translations in the choices sheet
                    if f"label:{language}" in choices_columns:
                        other_languages.append(language)

        # next, zip through the survey sheet to create a list of survey items
        output_data = {}
        group_stack = []
        groupless_count = 0
        last_group = ""
        for row in survey_data:
            if row[survey_columns['type']] and row[survey_columns['name']]:
                type_ = str(row[survey_columns['type']]).strip() if row[survey_columns['type']] is not None else ""
                name = str(row[survey_columns['name']]).strip() if row[survey_columns['name']] is not None else ""
                label = str(row[survey_columns['label']]).strip() if row[survey_columns['label']] is not None else ""
                hint = str(row[survey_columns['hint']]).strip() if row[survey_columns['hint']] is not None else ""
                
                if type_ == "begin group" or type_ == "begin repeat":
                    # new group, add to survey_data
                    output_data[name] = {
                        "module_name": name,
                        "module_title": label,
                        "module_intro": "",
                        "questions": {}
                    }
                    # and remember that we're in this group
                    group_stack.append(name)
                elif type_ == "end group" or type_ == "end repeat":
                    # pop the group stack to exit current group
                    group_stack.pop()
                elif label and type_ not in ["calculate", "calculate_here", "start", "end", "deviceid",
                                             "simserial", "phonenumber", "subscriberid", "caseid", "audio audit",
                                             "text audit", "speed violations count", "speed violations list",
                                             "speed violations audit"]:
                    # if question has a label isn't an invisible type, figure out which module it belongs to
                    if group_stack:
                        # group is most recent item added to the stack
                        module = group_stack[-1]
                    elif last_group and last_group.startswith("nomodule_"):
                        # we have a groupless module open and going, so continue with that
                        module = last_group
                    else:
                        # start a new groupless module
                        groupless_count += 1
                        module = f"nomodule_{groupless_count}"
                        # add new module to survey_data
                        output_data[module] = {
                            "module_name": module,
                            "module_title": "",
                            "module_intro": "",
                            "questions": {}
                        }
                    # remember this as the last group we were in
                    last_group = module

                    # assemble dictionary of questions
                    questions = {name: []}

                    # add item for base language first
                    item = {
                        "question": label,
                        "language": default_language,
                        "options": [],
                        "instructions": hint
                    }
                    # if the type is "select_one" or "select_multiple", add the choices
                    match = re.match(r"^(select_one|select_multiple) (.+)$", type_)
                    if match:
                        list_name = match.group(2)
                        # loop through all rows on the choices sheet with a matching list_name
                        for choice in choices_data:
                            if choice[choices_columns['list_name']] == list_name:
                                # add options one by one
                                item["options"].append({
                                    "label": str(choice[choices_columns['label']]).strip(),
                                    "value": str(choice[choices_columns['value']]).strip()
                                })
                    questions[name].append(item)

                    # next add translations (if any)
                    if other_languages:
                        for language in other_languages:
                            translated_label = str(row[survey_columns[f"label:{language}"]]).strip() \
                                if (f"label:{language}" in survey_columns
                                    and row[survey_columns[f"label:{language}"]] is not None) else ""
                            # only add translations with labels
                            if translated_label:
                                item = {
                                    "question": translated_label,
                                    "language": language,
                                    "options": [],
                                    "instructions": str(row[survey_columns[f"hint:{language}"]]).strip()
                                    if f"hint:{language}" in survey_columns
                                       and row[survey_columns[f"hint:{language}"]] is not None else ""
                                }
                                # if the type is "select_one" or "select_multiple", add the choices
                                match = re.match(r"^(select_one|select_multiple) (.+)$", type_)
                                if match:
                                    list_name = match.group(2)
                                    # loop through all rows on the choices sheet with a matching list_name
                                    for choice in choices_data:
                                        if choice[choices_columns['list_name']] == list_name:
                                            # add options one by one
                                            choice_label = str(choice[choices_columns[f"label:{language}"]]).strip() \
                                                if (f"label:{language}" in choices_columns and
                                                    choice[choices_columns[f"label:{language}"]] is not None) else ""
                                            if not choice_label:
                                                # use default language label if translated label missing
                                                choice_label = str(choice[choices_columns['label']]).strip()
                                            item["options"].append({
                                                "label": choice_label,
                                                "value": str(choice[choices_columns['value']]).strip()
                                                if f"hint:{language}" in choices_columns
                                                   and choice[choices_columns['value']] is not None else ""
                                            })
                                questions[name].append(item)

                    # update our output data with the new questions (translations for the current field)
                    output_data[module]["questions"].update(questions)

        # return structured data
        return clean_data(output_data)

    # otherwise, fallback to unstructured processing
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
        schema=ModuleList,
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
    :return: A list of strings for raw text content, or a dictionary for structured content (needs no extra parsing).
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
    # run through all modules
    for key, dt in data.items():
        # assemble module questions, dropping empty questions
        questions = {question: question_data for question, question_data in dt['questions'].items()
                     if question and question_data}

        # skip empty modules
        if not questions:
            parser_logger.log(logging.INFO, f"Skipping empty module: {key}")
        else:
            parser_logger.log(logging.INFO, f"* Module: {key}")
            # for each question, look for questions with the same language
            for question, question_data in questions.items():
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
                                    parser_logger.log(logging.INFO, f"  * Dropping duplicate question with "
                                                                    f"shorter content: "
                                                                    f"{question} - {question_data[j]['language']} - "
                                                                    f"{question_data[j]['question']} "
                                                                    f"({length_j} <= {length_i})")
                                else:
                                    to_remove.add(i)
                                    parser_logger.log(logging.INFO, f"  * Dropping duplicate question with "
                                                                    f"shorter content: "
                                                                    f"{question} - {question_data[i]['language']} - "
                                                                    f"{question_data[i]['question']} "
                                                                    f"({length_i} <= {length_j})")

                    # remove duplicates after identifying them
                    for index in sorted(to_remove, reverse=True):
                        question_data.pop(index)

            # add module to cleaned data
            cleaned_data[key] = {
                "module_name": dt['module_name'],
                "module_title": dt['module_title'],
                "module_intro": dt['module_intro'],
                "questions": questions
            }

    return cleaned_data


async def _invoke_task(func: Callable, param_dict: dict) -> dict:
    """
    Asynchronously invoke a function with parameters, capturing and returning any exceptions that occur.

    :param func: Task invocation function.
    :type func: Callable
    :param param_dict: Parameter dict to pass to the function.
    :type param_dict: dict
    :return: A dict with result ("success" or "error"), error (if result is "error"), and response (a dict).
    :rtype: dict
    """

    try:
        return {"result": "success", "error": None, "response": await func(param_dict)}
    except Exception as e:
        parser_logger.error(f"Task triggered error: {str(e)}")
        return {"result": "error", "error": f"Task triggered error: {str(e)}", "response": None}


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
    :return: A dict with modules (a dict with questions organized by module) and errors (a list of errors, if any).
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
    read_data = get_data_from_url(url)

    # if a dict is returned, that means the data was read as structured and we can return it straight away
    if isinstance(read_data, dict):
        return read_data

    # if it's an empty list, just return an empty dict
    if not read_data:
        return {}

    # otherwise, we have to process a content list
    # process list asynchronously, and track our LLM usage with an OpenAI callback
    with get_openai_callback() as cb:
        # create a list of tasks, then execute them asynchronously
        tasks = [_invoke_task(chain.ainvoke,
                              {"text": page, "examples": example_messages}) for page in read_data]
        results = await asyncio.gather(*tasks)

        # report LLM usage
        parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
        parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
        parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
        parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
        parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")

    # organize questions by module and question ID
    errors_output = []
    modules_output = {}
    question_module = {}
    unknown_module_count = 0
    unknown_id_count = 0
    for task_result in results:
        # if the task failed, add the error to the errors list and continue
        if task_result['result'] == 'error':
            errors_output.append(task_result['error'])
            continue

        for module in task_result['response'].modules:
            # construct module name from whatever details we have, defaulting to auto-naming as necessary
            if module.module_name == module.module_title:
                module_key = module.module_name
            elif module.module_name and module.module_title:
                module_key = f"{module.module_name} - {module.module_title}"
            elif module.module_name:
                module_key = module.module_name
            elif module.module_title:
                module_key = module.module_title
            else:
                unknown_module_count += 1
                module_key = f"MODULE_{unknown_module_count}"

            # run through all questions in module
            for question in module.questions:
                # process question, if any
                if question.question and question.question.strip():
                    # get question ID, if available
                    question_id = question.question_id.strip() if question.question_id else ''
                    if not question_id:
                        # if no question ID is provided, generate a unique ID
                        unknown_id_count += 1
                        question_id = f"question_{unknown_id_count}"

                    if question_id in question_module:
                        # if we've seen the question ID before, add this version to the same module as before
                        question_list = modules_output[question_module[question_id]].setdefault('questions', {})
                    else:
                        # otherwise, add it to the current module, adding the module to the output as needed
                        if module_key not in modules_output:
                            # add module to output, ignoring module title if same as the module name
                            modules_output[module_key] = {
                                "module_name": module.module_name if module.module_name else "",
                                "module_title": module.module_title if module.module_title
                                                                       and module.module_title != module.module_name
                                                                    else "",
                                "module_intro": module.module_intro if module.module_intro else "",
                                "questions": {}
                            }
                        question_list = modules_output[module_key].setdefault('questions', {})
                        question_module[question_id] = module_key

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
                    if question_id not in question_list:
                        question_list[question_id] = []
                    question_list[question_id].append({
                        'question': question.question,
                        'language': question.language if question.language else '',
                        'options': options,
                        'instructions': question.instructions if question.instructions else '',
                    })

    # return cleaned-up version of the data, along with any errors
    return {"modules": clean_data(modules_output), "errors": errors_output}


async def extract_data_from_directory(path_to_ingest: str, chain: LLMChain) -> list:
    """
    Extract structured data from all files in a specified directory.

    :param path_to_ingest: Path to the directory containing files to process.
    :type path_to_ingest: str
    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :return: A list of data extraction output from all processed files.
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


def _add_header_to_first_empty_cell(worksheet, header):
    for cell in worksheet[1]:
        if cell.value is None or not cell.value:
            cell.value = header
            break
    else:  # no break, meaning no empty cell was found
        max_column = worksheet.max_column
        worksheet.cell(row=1, column=max_column + 1, value=header)


def _get_columns_from_headers(worksheet, index_boost: int = 0) -> dict:
    return {cell.value: i + index_boost for i, cell in enumerate(worksheet[1])
            if cell.value is not None and cell.value.strip() != ''}


def output_parsed_data_to_xlsform(data: dict, form_id: str, form_title: str, output_file: str):
    """
    Output parsed data to an XLSForm file.

    :param data: Parsed data to output.
    :type data: dict
    :param form_id: Form ID to set in the XLSForm file.
    :type form_id: str
    :param form_title: Form title to set in the XLSForm file.
    :type form_title: str
    :param output_file: Path to the output XLSForm file.
    :type output_file: str
    """

    # load the workbook
    wb = load_workbook(empty_form_path)
    survey_ws = wb['survey']
    survey_columns = _get_columns_from_headers(survey_ws, 1)
    choices_ws = wb['choices']
    choices_columns = _get_columns_from_headers(choices_ws, 1)
    settings_ws = wb['settings']
    settings_columns = _get_columns_from_headers(settings_ws, 1)

    # set the 'form_title' and 'form_id' values in row 2
    settings_ws.cell(row=2, column=settings_columns['form_id'], value=form_id)
    settings_ws.cell(row=2, column=settings_columns['form_title'], value=form_title)

    # initialize our row counters to start adding at the second row
    survey_row_counter = 2
    choices_row_counter = 2

    # iterate over each module in the data
    primary_language = ""
    for module in data.values():
        # create a safe version of the module name by replacing anything not a digit or a letter with an _
        safe_module_name = re.sub(r'\W+', '_', module['module_name']).lower()

        # open a group for the module
        survey_ws.cell(row=survey_row_counter, column=survey_columns['type'], value='begin group')
        survey_ws.cell(row=survey_row_counter, column=survey_columns['name'], value=safe_module_name)
        if module['module_title']:
            survey_ws.cell(row=survey_row_counter, column=survey_columns['label'], value=module['module_title'])
        survey_row_counter += 1

        # if there is a 'module_intro' value in the module, add it as a note field
        if module['module_intro']:
            survey_ws.cell(row=survey_row_counter, column=survey_columns['type'], value='note')
            survey_ws.cell(row=survey_row_counter, column=survey_columns['label'], value=module['module_intro'])
            survey_row_counter += 1

        # iterate over each question in the module
        for question_id, question_data in module['questions'].items():
            # create a safe version of the question ID by replacing anything not a digit or a letter with an _
            safe_question_id = re.sub(r'\W+', '_', question_id).lower()
            # if safe_question_id doesn't begin with a letter, put a 'q' on the front
            if not safe_question_id[0].isalpha():
                safe_question_id = 'q' + safe_question_id

            # iterate over each translation for the question
            has_choices = False
            for translation in question_data:
                # if we don't have our primary language yet, use the first language we find
                if not primary_language:
                    primary_language = translation['language']
                    settings_ws.cell(row=2, column=settings_columns['default_language'], value=primary_language)

                # set language suffix
                if translation['language'] and translation['language'] != primary_language:
                    language_suffix = f":{translation['language']}"
                    label_column = 'label' + language_suffix
                    hint_column = 'hint' + language_suffix
                    # also add translation columns as necessary
                    if label_column not in survey_columns:
                        _add_header_to_first_empty_cell(survey_ws, label_column)
                        survey_columns = _get_columns_from_headers(survey_ws, 1)
                    if translation['options'] and label_column not in choices_columns:
                        _add_header_to_first_empty_cell(choices_ws, label_column)
                        choices_columns = _get_columns_from_headers(choices_ws, 1)
                    if hint_column not in survey_columns:
                        _add_header_to_first_empty_cell(survey_ws, hint_column)
                        survey_columns = _get_columns_from_headers(survey_ws, 1)
                else:
                    label_column = 'label'
                    hint_column = 'hint'
                    # if there are options, we'll need to add them to the choices sheet (all at the end)
                    #   (we let the primary language govern whether there are choices)
                    if translation['options']:
                        has_choices = True

                # add the question to the survey sheet
                survey_ws.cell(row=survey_row_counter, column=survey_columns['type'],
                               value='text' if not has_choices else 'select_one ' + safe_question_id)
                survey_ws.cell(row=survey_row_counter, column=survey_columns['name'], value=safe_question_id)
                survey_ws.cell(row=survey_row_counter, column=survey_columns[label_column],
                               value=translation['question'])
                survey_ws.cell(row=survey_row_counter, column=survey_columns[hint_column],
                               value=translation['instructions'])

            # increment survey row when we're done adding all translations
            survey_row_counter += 1

            # then, finally, add all choice options (with all translations)
            if has_choices:
                values_added = []
                for translation in question_data:
                    for option in translation['options']:
                        if option['value'] not in values_added:
                            # if we haven't added this option value yet, add it for all translations
                            choices_ws.cell(row=choices_row_counter, column=choices_columns['list_name'],
                                            value=safe_question_id)
                            choices_ws.cell(row=choices_row_counter, column=choices_columns['value'],
                                            value=option['value'])
                            for inner_translation in question_data:
                                for inner_option in inner_translation['options']:
                                    if inner_option['value'] == option['value']:
                                        if (inner_translation['language'] and inner_translation['language'] !=
                                                primary_language):
                                            label_column = 'label' + f":{inner_translation['language']}"
                                        else:
                                            label_column = 'label'
                                        choices_ws.cell(row=choices_row_counter, column=choices_columns[label_column],
                                                        value=inner_option['label'])
                            choices_row_counter += 1
                            values_added.append(option['value'])

        # close the module's group
        survey_ws.cell(row=survey_row_counter, column=survey_columns['type'], value='end group')
        survey_ws.cell(row=survey_row_counter, column=survey_columns['name'], value=safe_module_name)
        survey_row_counter += 1

    # Save the workbook to the specified output file
    wb.save(output_file)
