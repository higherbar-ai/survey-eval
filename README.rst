==========
surveyeval
==========

This repository contains a toolkit for AI-powered survey instrument evaluation. It's still in early development, but 
is ready to support piloting and experimentation. To learn more about the overall project, see 
`this blog post <https://www.linkedin.com/pulse/under-the-hood-ai-beyond-chatbots-christopher-robert-dquue>`_.

Installation
------------

Installing the latest version with pip::

    pip install surveyeval

Note that you might need to install additional requirements to use the ``survey_parser`` module. Only requirements for
the core evaluation engine are automatically installed by ``pip``. To install all requirements, use
`the requirements list from the full repo <https://github.com/higherbar-ai/survey-eval/blob/main/requirements.txt>`_::

    pip install -r requirements.txt

Overview
---------

Here are the basics:

#. This toolkit includes code to read, parse, and evaluate survey instruments.
#. The ``file-evaluation-example.ipynb`` Jupyter workbook provides a working example for evaluating a single survey
   instrument file. It includes details on how to install, configure, and run.
#. The evaluation engine itself lives in the ``evaluation_engine`` module. It provides a pretty basic framework for
   applying different evaluation lenses to a survey instrument.
#. The ``core_evaluation_lenses`` module contains an initial set of evaluation lenses that can be applied to survey
   instruments. These are the ones applied in the example workbook. They are:

   a. ``PhrasingEvaluationLens``: Cases where phrasing might be adjusted to improve respondent understanding and reduce
      measurement error (i.e., the kinds of phrasing issues that would be identified through rigorous cognitive
      interviewing or other forms of validation)
   b. ``TranslationEvaluationLens``: Cases where translations are inaccurate or phrased such that they might lead to
      differing response patterns
   c. ``BiasEvaluationLens``: Cases where phrasing might be improved to remove implicit bias or stigmatizing language
      (inspired by `this very helpful post <https://www.linkedin.com/pulse/using-chatgpt-counter-bias-prejudice-discrimination-johannes-schunter/>`_
      on the subject of using ChatGPT to identify bias)
   d. ``ValidatedInstrumentEvaluationLens``: Cases where a validated instrument might be adapted to better measure an
      inferred construct of interest
#. The code for reading and parsing files is in the ``survey_parser`` module. There's much there that can be improved
   about how different file formats are read into raw text, and then how they're parsed into questions, modules, and so 
   on. In particular, one might improve the range of examples provided to the LLM.

You can run the ``file-evaluation-example.ipynb`` workbook as-is, but you might also consider customizing the
core evaluation lenses to better meet your needs and/or adding your own evaluation lenses to the workbook. When adding
new lenses, you can just use any of the initial lenses as a template.

If you make use of this toolkit, we'd love to hear from you — and help to share your results with the community. Please
email us at ``info@higherbar.ai``.

Technical notes
---------------

Reading input files
^^^^^^^^^^^^^^^^^^^

The ``survey_parser`` module contains code for reading input files. It currently supports the following
file formats:

1. ``.docx``: Word files are read in the ``read_docx()`` function, using LangChain's ``UnstructuredFileLoader()`` function.
   Note that text within "content controls" is effectively invisible, so you might need to open your file, select all, 
   and select "Remove content controls" to render the text visible 
   (`see here for more on content controls <https://learn.microsoft.com/en-us/office/client-developer/word/content-controls-in-word>`_).
2. ``.pdf``: PDF files are read in the ``read_pdf_combined()`` function, which tries to read the text and tables in a PDF
   (separately), combine them together, and then fall back to using an OCR reader if that process didn't find much 
   text. There is a ton of room for improvement here.
3. ``.xlsx``: Excel files are read in the ``parse_xlsx()`` function, in two ways. If the file looks like it's in
   `XLSForm format <https://xlsform.org/en/>`_, it parses it accordingly; this parsing should be completely lossless
   and requires no additional parsing at later stages. If the file does not appear to be an XLSForm, the reader falls
   back to using LangChain's ``UnstructuredExcelLoader()`` to load the workbook in HTML format, then uses that HTML as
   the raw text for parsing. XLSForm handling should be robust, but there is much that can be improved in how other
   formats are handled.
4. ``.csv``: CSV files are read in the ``parse_csv()`` function, also in two ways. If the file looks like a REDCap
   data dictionary, it will parse the columns accordingly (requiring little to no later processing). Otherwise, it
   falls back to just reading the file as raw text. There is much that can be improved here, particularly in how
   REDCap data dictionaries are handled (e.g., the current approach doesn't handle modules or translations).
5. ``.html``: HTML files are read in the ``read_html()`` function, then converted into markdown for parsing.

All of the raw content is split into 3,000-character chunks with 500 characters of overlap, before being passed on
for parsing. This is necessary to (a) avoid overflowing LLM context windows, (b) avoid overflowing output token
limits, and (c) allow the LLM to focus on a tractable amount of text in any given request (with the latter becoming
more important as the constraints on context windows are relaxed).

Overall, the code for reading files performs pretty poorly for all but the simplest formats. There's much work to do
here to improve quality.

Parsing input files
^^^^^^^^^^^^^^^^^^^

The actual parsing happens in the ``survey_parser`` module, with LLM assistance via
`the LangChain approach to extraction <https://python.langchain.com/docs/use_cases/extraction/>`_.

If performance is poor for your file, you can try giving the parser some examples from the raw data read from your
file. Search for ``examples`` in the ``survey_parser`` module to see the baseline examples. Then create your own
examples and pass them in as the ``replacement_examples`` or ``additional_examples`` parameters to the
``extract_data()`` function. This will help the LLM to better understand your file format.

Tracking and reporting costs
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The API usage — including cost estimates — is currently output to the console as INFO-level logs, but only for the
parsing stage. The evaluation stage doesn't currently track or report costs.

Roadmap
-------

There's much that can be improved here. For example:

* We should track and report costs for the evaluation stage of the process.
* We should generally overhaul the ``survey_parser`` module to better ingest different file formats into
  raw text that works consistently well for parsing. Better PDF and REDCap support, in particular, would be
  nice.
* We should add an LLM cache that avoids calling out to the LLM for responses that it already has from prior requests.
  After all, it's common to evaluate the same instrument multiple times, and it's incredibly wasteful to 
  keep going back to the LLM for the same responses every time (for requests that haven't changed in any way).
* We should improve how findings are scored and filtered, to avoid giving overwhelming numbers of minor 
  recommendations.
* We should improve the output format to be more user-friendly. (For example, a direct Word output with comments and 
  tracked changes would be very nice).
* We should add more evaluation lenses. For example:
  * Double-barreled questions: Does any question ask about two things at once?
  * Leading questions: Are questions neutral and don’t lead the respondent towards a particular answer?
  * Response options: Are the response options exhaustive and mutually exclusive?
  * Question order effects: The order in which questions appear can influence how respondents interpret and answer subsequent items. It's essential to evaluate if any questions might be leading or priming respondents in a way that could bias their subsequent answers.
  * Consistency: Are scales used consistently throughout the survey?
  * Reliability and validity: If established scales are used, have they been validated for the target population?
  * Length and respondent burden: Is the survey too long? Long surveys can lead to respondent fatigue, which in turn might lead to decreased accuracy or increased drop-out rates.
* Ideally, we would parse modules into logical sub-modules that appear to measure a single construct, so that we can
  better evaluate whether to recommend adaptation of validated instruments. Right now, an entire module is evaluated
  at once, but modules often contain measurement of multiple constructs.

Credits
-------

This toolkit was originally developed by `Higher Bar AI <https://higherbar.ai>`_, a public benefit corporation, with
generous support from `Dobility, the makers of SurveyCTO <https://surveycto.com>`_.

Full documentation
------------------

See the full reference documentation here:

    https://surveyeval.readthedocs.io/

Local development
-----------------

To develop locally:

#. ``git clone https://github.com/higherbar-ai/survey-eval``
#. ``cd survey-eval``
#. ``python -m venv venv``
#. ``source venv/bin/activate``
#. ``pip install -r requirements.txt``

For convenience, the repo includes ``.idea`` project files for PyCharm.

To rebuild the documentation:

#. Update version number in ``/docs/source/conf.py``
#. Update layout or options as needed in ``/docs/source/index.rst``
#. In a terminal window, from the project directory:
    a. ``cd docs``
    b. ``SPHINX_APIDOC_OPTIONS=members,show-inheritance sphinx-apidoc -o source ../src/surveyeval --separate --force``
    c. ``make clean html``

To rebuild the distribution packages:

#. For the PyPI package:
    a. Update version number (and any build options) in ``/setup.py``
    b. Confirm credentials and settings in ``~/.pypirc``
    c. Run ``/setup.py`` for the ``bdist_wheel`` and ``sdist`` build types (*Tools... Run setup.py task...* in PyCharm)
    d. Delete old builds from ``/dist``
    e. In a terminal window:
        i. ``twine upload dist/* --verbose``
#. For GitHub:
    a. Commit everything to GitHub and merge to ``main`` branch
    b. Add new release, linking to new tag like ``v#.#.#`` in main branch
#. For readthedocs.io:
    a. Go to https://readthedocs.org/projects/surveyeval/, log in, and click to rebuild from GitHub (only if it
       doesn't automatically trigger)
