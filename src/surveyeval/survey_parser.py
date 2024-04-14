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

from typing import Optional
import os
import tempfile
import logging
import json
import asyncio
import requests
from urllib.parse import urlparse
from pydantic import BaseModel, Field
from kor.extraction import create_extraction_chain
from kor import from_pydantic
from kor.nodes import Object
from kor.validators import Validator
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import PromptTemplate
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
from kor.documents.html import MarkdownifyHTMLProcessor
from langchain.schema import Document as SchemaDocument
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredFileLoader, UnstructuredExcelLoader

# initialize global resources
parser_logger = logging.getLogger(__name__)
nltk.download('punkt')
nlp = spacy.load('en_core_web_sm')

# initialize global variables
langchain_splitter_chunk_size = 6000
langchain_splitter_overlap_size = 500


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
                    question_data['options'] = [{"value": c.split(',')[0].strip(), "label": c.split(',')[1].strip()} for
                                                c in choices]
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


def create_schema(question_id_spec: str = None, module_spec: str = None, module_desc_spec: str = None,
                  question_spec: str = None,
                  instructions_spec: str = None, options_spec: str = None, language_spec: str = None,
                  kor_general_spec: str = None) -> tuple[Object, Validator]:
    """
    Create a schema based on a pydantic model for questionnaire data.

    This function generates a schema using dynamic descriptions for the fields of a Question class,
    and returns a schema and an extraction validator.

    :param question_id_spec: Specification for the 'question_id' field.
    :type question_id_spec: str
    :param module_spec: Specification for the 'module' field.
    :type module_spec: str
    :param module_desc_spec: Specification for the 'module_description' field.
    :type module_desc_spec: str
    :param question_spec: Specification for the 'question' field.
    :type question_spec: str
    :param instructions_spec: Specification for the 'instructions' field.
    :type instructions_spec: str
    :param options_spec: Specification for the 'options' field.
    :type options_spec: str
    :param language_spec: Specification for the 'language' field.
    :type language_spec: str
    :param kor_general_spec: Overall specification for the schema.
    :type kor_general_spec: str
    :return: A tuple containing the schema and extraction validator.
    :rtype: tuple[Object, Validator]
    """

    # set defaults as needed
    if not question_id_spec:
        question_id_spec = ('Question ID: a numeric or alphanumeric identifier or short variable name identifying a '
                            'specific question, usually located just before or at the beginning of a new question.')
    if not module_spec:
        module_spec = ('Module title: Represents the main section or category within which a series of questions are '
                       'located (e.g., "Health" or "Demographics"). It might include a number or index, but should '
                       'also include a short title.')
    if not module_desc_spec:
        module_desc_spec = ("Module introduction: Introductory text or instructions that appear at the start of a new "
                            "module, before the module's questions appear.")
    if not question_spec:
        question_spec = ('Question: A single question or label/description of a single form field, often following '
                         'a numerical code or identifier like "2.01." or "gender:" Must be text designed to elicit '
                         'specific information, often in the form of a question (e.g., "How old are you?") or prompt '
                         '(e.g., "Your age:"). Might be in different languages, but the structure remains the same.')
    if not instructions_spec:
        instructions_spec = ('Question instructions: Instructions or other guidance about how to ask or answer the '
                             'question, including enumerator or interviewer instructions. If the question includes '
                             'a list of specific response options, do NOT include those in the instructions.')
    if not options_spec:
        options_spec = ("Question options: The list of specific response options for multiple-choice questions. "
                        "Often listed immediately after the question or instructions. Might include numbers, "
                        "letters, or specific codes followed by the actual response option text. Separate options "
                        "with a space, a pipe symbol, and another space, like this: '1. Yes | 2. No'.")
    if not language_spec:
        language_spec = 'Question language: The language in which the question is written.'
    if not kor_general_spec:
        kor_general_spec = ('Questionnaire: A questionnaire consists of a list of questions or prompts (question) '
                            'that are used to collect data from respondents. Each question might include a short ID '
                            'number or name (question_id), instructions, and/or a list of specific response options '
                            '(options), and each question might appear in multiple languages (language). These '
                            'questions might be organized within a series of modules (or sections), each of which '
                            'might have a title and introductory instructions '
                            '(module_description). You must return the questionnaire in the '
                            'same order as it was given to you and in each json you must return either a module '
                            'or question. If there is a question that is not complete, DO NOT return it.')

    class QuestionnaireData(BaseModel):
        """
        Pydantic model representing a questionnaire question.

        Each field of the model is optional and includes a description provided
        as a parameter to the create_schema function.
        """

        question_id: Optional[str] = Field(description=question_id_spec)
        module: Optional[str] = Field(description=module_spec)
        module_description: Optional[str] = Field(description=module_desc_spec)
        question: Optional[str] = Field(description=question_spec)
        instructions: Optional[str] = Field(description=instructions_spec)
        options: Optional[str] = Field(description=options_spec)
        language: Optional[str] = Field(description=language_spec)

    # generate schema and extraction validator from the QuestionnaireData class
    schema, extraction_validator = from_pydantic(
        QuestionnaireData,
        description=kor_general_spec,
        examples=[
            ("""2. Demographics

We’ll begin with some questions so that we can get to know you and your family.

[BIRTHYR] What year were you born?

[GENDER] Which gender do you identify with?

Female

Male

Non-binary

Prefer not to answer

[ZIPCODE] What is your zip code?""", {"questionnairedata": [
                {
                    "module": "2. Demographics",
                    "module_description": "We’ll begin with some questions so that we can get to know you and your "
                                          "family.",
                },
                {
                    "question_id": "BIRTHYR",
                    "question": "What year were you born?",
                    "language": "English"
                },
                {
                    "question_id": "GENDER",
                    "question": "Which gender do you identify with?",
                    "options": "Female | Male | Non-binary | Prefer not to answer",
                    "language": "English"
                },
                {
                    "question_id": "ZIPCODE",
                    "question": "What is your zip code?",
                    "language": "English"
                }
            ]}),
            ("""[EDUCATION] What is the highest level of education you have completed?

o Less than high school

o High school / GED

o Some college

o 2-year college degree

o 4-year college degree

o Vocational training

o Graduate degree

o Prefer not to answer""", {"questionnairedata": [
                {
                    "question_id": "EDUCATION",
                    "question": "What is the highest level of education you have completed?",
                    "options": "Less than high school | High school / GED | Some college | 2-year college degree | "
                               "4-year college degree | Vocational training | Graduate degree | Prefer not to answer",
                    "language": "English"
                }
            ]}),
            ("""And how much do you disagree or agree with the following statements? For each statement, please """
             """rate how much the pair of traits applies to you, even if one trait applies more strongly than the """
             """other. I see myself as... [For each: 1=Strongly disagree, 7=Strongly agree]

[BIG5Q1] Extraverted, enthusiastic

[BIG5Q2] Critical, quarrelsome

[BIG5Q3] Dependable, self-disciplined""", {"questionnairedata": [
                {
                    "question_id": "BIG5Q1",
                    "question": "And how much do you disagree or agree with the following statement? I see myself "
                                "as... Extraverted, enthusiastic",
                    "options": "1=Strongly disagree | 2 | 3 | 4 | 5 | 6 | 7=Strongly agree",
                    "instructions": "Please rate how much the pair of traits applies to you, even if one trait applies "
                                    "more strongly than the other.",
                    "language": "English"
                },
                {
                    "question_id": "BIG5Q2",
                    "question": "And how much do you disagree or agree with the following statement? I see myself "
                                "as... Critical, quarrelsome",
                    "options": "1=Strongly disagree | 2 | 3 | 4 | 5 | 6 | 7=Strongly agree",
                    "instructions": "Please rate how much the pair of traits applies to you, even if one trait applies "
                                    "more strongly than the other.",
                    "language": "English"
                }]}),
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

3. Do you contribute the same amount or more to your savings each month?""", {"questionnairedata": [
                    {
                        "module": "4. Savings Habits",
                        "module_description": "Next, we will ask questions over your monthly saving habits and the "
                                              "potential methods that are used to save money.",
                    },
                    {
                        "question_id": "1",
                        "question": "On average, how much money do you spend monthly on essential goods below that "
                                    "contribute to your wellbeing (explain/ add in an example)",
                        "language": "English"
                    },
                    {
                        "question_id": "2",
                        "question": "How do you typically spend your monthly income?",
                        "options": "a. Home and Housing | b. Retirement | c. Bills and Utility | d. Medical "
                                   "(Physical and Mental Treatment and Care) | e. Taxes | f. Insurance | g. Credit "
                                   "Card Payments (if applicable) | h. Food | i. Shopping and personal items | j. "
                                      "Other | k. I am not able to save money each month | l. Nothing | m. Don’t Know",
                        "instructions": "(choose all that may apply)",
                        "language": "English"
                    },
                    {
                        "question_id": "3",
                        "question": "Do you contribute the same amount or more to your savings each month?",
                        "language": "English"
                    }
                ]}
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
</tr>""", {"questionnairedata": [
                    {
                        "module": "CONS. Introduction and Consent",
                    },
                    {
                        "question": """Good morning/afternoon/evening. My name is ______________________ from """
                        """Innovations from Poverty Action, a """
                        """Mexican research NGO. \n \n We would like to invite you to participate in a survey """
                        """lasting about 20 minutes about the effects of covid-19 on economic and social """
                        """conditions in the Mexico City metropolitan area. If you are eligible for the survey """
                        """we will compensate you [30 pesos] in airtime for completing your survey.""",
                        "language": "English"
                    },
                    {
                        "question_id": "cons1",
                        "question": "Can I give you more information?",
                        "options": "Y | N",
                        "instructions": "*If cons1=N\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    },
                    {
                        "question_id": "end4",
                        "question": "What is your first name?",
                        "language": "English"
                    },
                    {
                        "question_id": "dem1",
                        "question": "How old are you?",
                        "instructions": "*Enter age*\n ###\n"
                                        "*If DEM1&lt;18*\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    }
                ]}
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
</tr>""", {"questionnairedata": [
                    {
                        "question_id": "inc11_mex",
                        "question": """If schools and daycares remained closed and workplaces re-opened, would """
                        """anyone in your household have to stay home and not return to work in order to care for """
                        """children too young to stay at home without supervision?""",
                        "options": "Grandparents | Hired babysitter | Neighbors | Mother who normally st | Mother who "
                                   "normally works outside the home | Father who normally works outside the home | "
                                   "Older sibling | DNK",
                        "instructions": "*If YES to INC12_mex*\n*Read out, select multiple possible*",
                        "language": "English"
                    },
                    {
                        "module": "NET. Social Safety Net",
                    },
                    {
                        "question_id": "net1",
                        "question": """Do you usually receive a regular transfer from any cash transfer or other """
                                    """in-kind social support program?\n \n HINT: Social safety net programs """
                                    """include cash transfers and in-kind food transfers (food stamps and vouchers, """
                                    """food rations, and emergency food distribution). Example includes XXX cash """
                                    """transfer programme.""",
                        "options": "Y | N | DNK",
                        "instructions": "*If cons1=N\n Thank you for your response. We will end the survey now. "
                                        "[End survey]",
                        "language": "English"
                    }
                ]}
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
</tr>""", {"questionnairedata": [
                    {
                        "module": "POL. POLICING",
                        "module_description": "Now I am going to ask you some questions about the main problems of "
                                              "insecurity in Mexico City and the performance of the city police since "
                                              "the coronavirus pandemic began around March 20, 2020."
                    },
                    {
                        "question_id": "POL1",
                        "question": """Compared to the level of insecurity that existed in your neighborhood before """
                                    """the pandemic began, do you consider that the level of insecurity in your """
                                    """neighborhood decreased, remained more or less the same, or increased?""",
                        "options": "Decreased | it was more or less the same | increased | (777) Doesn’t answer | "
                                   "(888) Doesn’t know | (999) Doesn’t apply",
                        "language": "English"
                    }
                ]}
            )
        ],
        many=True,
    )

    return schema, extraction_validator


def generate_extractor_chain(model_input: str, api_base: str, openai_api_key: str, open_api_version: str,
                             schema: Object, default_kor_prompt: str = None, provider: str = "azure") -> LLMChain:
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
    :param schema: Schema definition for the extraction.
    :type schema: Object
    :param default_kor_prompt: Default prompt template for knowledge ordering and reasoning.
    :type default_kor_prompt: str
    :param provider: Provider for the LLM service ("openai" for direct OpenAI, "azure" for Azure). Default is "azure".
    :type provider: str
    :return: An extraction chain configured with the specified parameters.
    :rtype: LLMChain
    """

    # set defaults as needed
    if not default_kor_prompt:
        default_kor_prompt = ("Your goal is to extract structured information from the user's input that matches "
                              "the format described below. When extracting information, please make sure it matches "
                              "the type information exactly. Please return the information in order. Only extract the "
                              "information that is in the document. Do not add any extra information. Do not add any "
                              "attributes that do not appear in the schema shown below.\n\n"
                              "{type_description}\n\n{format_instructions}\n\n")

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

    # define the prompt template for the extraction chain
    template = PromptTemplate(
        input_variables=["type_description", "format_instructions"],
        template=default_kor_prompt,
    )

    # create and return the extraction chain
    chain = create_extraction_chain(llm, schema, encoder_or_encoder_class="JSON",
                                    instruction_template=template, input_formatter="triple_quotes")
    return chain


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


def process_kor_data(data: dict) -> list:
    """
    Process and structure data specifically for knowledge ordering and reasoning.

    :param data: Data to be processed, expected to have a 'questionnairedata' field.
    :type data: dict
    :return: A list of structured data.
    :rtype: list
    """

    # log raw data for debugging purposes
    parser_logger.log(logging.DEBUG, f"Raw data:\n{json.dumps(data, indent=2)}")

    # handle either 1 or 2 layers of outer nesting
    inner_data = data.get('questionnairedata', [])
    if len(inner_data) == 1 and isinstance(inner_data[0], dict) and 'questionnairedata' in inner_data[0]:
        # if the data is nested, process the inner data
        inner_data = inner_data[0]['questionnairedata']

    return [record for record in inner_data]


async def safe_apredict(chain: LLMChain, page: str):
    """
    Asynchronously predict with error handling, ensuring a safe call to the AI prediction chain.

    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :param page: The input text to be processed.
    :type page: str
    :return: The prediction result or a default value in case of an error.
    """

    try:
        return await chain.apredict(text=page)
    except Exception as e:
        # report, then return a default value in case of an error
        parser_logger.log(logging.ERROR, f"An error occurred: {e}")
        return {'data': []}


def total_string_length(d: dict) -> int:
    """
    Calculate the total string length of all values in a dictionary.

    :param d: The dictionary whose values' string lengths are to be summed.
    :type d: dict
    :return: The total string length of all values.
    :rtype: int
    """

    return sum(len(str(value)) for value in d.values())


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
            # for each question, look for questions with the same question text
            for question, question_data in module.items():
                if question_data:
                    to_remove = set()
                    for i in range(len(question_data)):
                        for j in range(i + 1, len(question_data)):
                            if question_data[i]['question'] == question_data[j]['question']:
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

                    # Remove duplicates after identifying them
                    for index in sorted(to_remove, reverse=True):
                        question_data.pop(index)

            cleaned_data[key] = module

    return cleaned_data


async def extract_data(chain: LLMChain, url: str) -> dict:
    """
    Asynchronously process the content from a given URL and parse it into structured data.

    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :param url: URL of the document to process.
    :type url: str
    :return: Structured and cleaned data.
    :rtype: dict
    """

    docs = get_data_from_url(url)
    structured = []
    grouped_content = {}

    if isinstance(docs, list):
        # process list of documents asynchronously
        if docs:
            # track our LLM usage with an OpenAI callback
            with get_openai_callback() as cb:
                # create a list of tasks, then execute them asynchronously
                tasks = [safe_apredict(chain, page) for page in docs]
                results = await asyncio.gather(*tasks)

                # report LLM usage
                parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
                parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
                parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
                parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
                parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")

                # parse list of results
                for res in results:
                    # if the resulting data is a dict, process it
                    if isinstance(res['data'], dict):
                        structured.extend(process_kor_data(res['data']))
    else:
        # process single document, tracking our LLM usage with an OpenAI callback
        with get_openai_callback() as cb:
            structured = process_kor_data(docs)

            # report LLM usage
            parser_logger.log(logging.INFO, f"Tokens consumed:: {cb.total_tokens}")
            parser_logger.log(logging.INFO, f"  Prompt tokens: {cb.prompt_tokens}")
            parser_logger.log(logging.INFO, f"  Completion tokens: {cb.completion_tokens}")
            parser_logger.log(logging.INFO, f"Successful Requests: {cb.successful_requests}")
            parser_logger.log(logging.INFO, f"Cost: ${cb.total_cost}")

    # organize questions by module and question ID
    question_module = {}
    current_module = '(none)'
    unknown_id_count = 0
    for record in structured:
        # get module name, defaulting to the current one
        module = record.get('module', current_module)

        # process question, if any
        if record.get('question', '').strip():
            # get question ID, if available
            question_id = record.get('question_id', '')
            if not question_id:
                # if no question ID is provided, generate a unique ID
                unknown_id_count += 1
                question_id = f"unknown_id_{unknown_id_count}"

            # always keep questions with the same ID together in the same module
            if question_id in question_module:
                module = question_module[question_id]
            else:
                question_module[question_id] = module

            # add question, grouped by module and question ID
            grouped_content.setdefault(module, {}).setdefault(question_id, [])
            grouped_content[module][question_id].append({
                'question': record['question'],
                'language': record.get('language', ''),
                'options': record.get('options', ''),
                'instructions': record.get('instructions', ''),
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


async def extract_data_from_file(file_path: str, chain: LLMChain) -> dict:
    """
    Extract structured data from a single file.

    :param file_path: Path to the file to process.
    :type file_path: str
    :param chain: The AI prediction chain to be used.
    :type chain: LLMChain
    :return: Structured data extracted from the processed file.
    :rtype: dict
    """

    return await extract_data(chain, file_path)
