==========
surveyeval
==========

The ``surveyeval`` package is a toolkit for AI-powered survey instrument evaluation. It's still in early development,
but is ready to support piloting and experimentation. To learn more about the overall project, see
`this blog post <https://www.linkedin.com/pulse/under-the-hood-ai-beyond-chatbots-christopher-robert-dquue>`_.

Installation
------------

Install the full version with pip::

    pip install surveyeval[parser]

If you don't need anything in the ``survey_parser`` module (relating to reading, parsing, and converting
survey files), you can install a slimmed-down version with::

    pip install surveyeval

Additional document-parsing dependencies
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

If you installed the full version with survey-parsing capabilities (``surveyeval[parsing]``), you'll also need
to install several other dependencies, which you can do by running the
`initial-setup.ipynb <https://github.com/higherbar-ai/survey-eval/blob/main/src/initial-setup.ipynb>`_ Jupyter
notebook — or by installing them manually as follows.

First, download NTLK data for natural language text processing::

    # download NLTK data
    import nltk
    nltk.download('punkt', force=True)

Then install ``libreoffice`` for converting Office documents to PDF.

  On Linux::

    # install LibreOffice for document processing
    !apt-get install -y libreoffice

  On MacOS::

    # install LibreOffice for document processing
    brew install libreoffice

  On Windows::

    # install LibreOffice for document processing
    choco install -y libreoffice

AWS Bedrock support
^^^^^^^^^^^^^^^^^^^

Finally, if you're accessing models via AWS Bedrock, the AWS CLI needs to be installed and configured for AWS access.

Jupyter notebooks with Google Colab support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

You can use `the colab-or-not package <https://github.com/higherbar-ai/colab-or-not>`_ to initialize a Jupyter notebook
for Google Colab or other environments::

    %pip install colab-or-not surveyeval

    # download NLTK data
    import nltk
    nltk.download('punkt', force=True)

    # set up our notebook environment (including LibreOffice)
    from colab_or_not import NotebookBridge
    notebook_env = NotebookBridge(
        system_packages=["libreoffice"],
        config_path="~/.hbai/survey-eval.env",
        config_template={
            "openai_api_key": "",
            "openai_model": "",
            "azure_api_key": "",
            "azure_api_base": "",
            "azure_api_engine": "",
            "azure_api_version": "",
            "anthropic_api_key": "",
            "anthropic_model": "",
            "langsmith_api_key": "",
        }
    )
    notebook_env.setup_environment()

See `file-evaluation-example.ipynb <https://github.com/higherbar-ai/survey-eval/blob/main/src/file-evaluation-example.ipynb>`_
for an example.

Overview
---------

Here are the basics:

#. This toolkit includes code to read, parse, and evaluate survey instruments.
#. `The file-evaluation-example.ipynb Jupyter notebook <https://github.com/higherbar-ai/survey-eval/blob/main/src/file-evaluation-example.ipynb>`_
   provides a working example for evaluating a single survey instrument file. It includes details on how to install,
   configure, and run.
#. The evaluation engine itself lives in the ``evaluation_engine`` module. It provides a pretty basic framework for
   applying different evaluation lenses to a survey instrument.
#. The ``core_evaluation_lenses`` module contains an initial set of evaluation lenses that can be applied to survey
   instruments. These are the ones applied in the example notebook. They are:

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
#. The code for reading and parsing files is in the ``survey_parser`` module. Aside from
   `XLSForm <https://xlsform.org/en/>`_ files and REDCap data dictionaries — which are parsed directly — the module
   relies heavily on
   `the ai_workflows package <https://github.com/higherbar-ai/ai-workflows>`_ for reading files and using an LLM to
   assist with parsing.

You can run the
`file-evaluation-example.ipynb <https://github.com/higherbar-ai/survey-eval/blob/main/src/file-evaluation-example.ipynb>`_
notebook as-is, but you might also consider customizing the core evaluation lenses to better meet your needs and/or
adding your own evaluation lenses to the notebook. When adding new lenses, you can just use any of the initial lenses
as a template.

If you make use of this toolkit, we'd love to hear from you — and help to share your results with the community. Please
email us at ``info@higherbar.ai``.

Technical notes
---------------

Reading and parsing input files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The ``survey_parser`` module contains code for reading input files. It directly supports two popular formats for
digital instruments (`XLSForm <https://xlsform.org/en/>`_ files and REDCap data dictionaries), which are read straight
into a structured format that is ready for evaluation. A wide variety of other document formats are supported via the
`ai_workflows <https://github.com/higherbar-ai/ai-workflows>`_ package, in two stages:

1. In the first stage, raw text is extracted from the document in a basic Markdown format. The techniques used depend
   on the file format, but when possible an LLM is used to transform each page into Markdown text, and then all of the
   text is merged together. LLM-based extraction can be slow and expensive (roughly $0.015/page), so you can disable it
   by setting the ``use_llm`` parameter to ``False`` when calling the ``read_survey_contents()`` function. For example::

    from surveyeval.survey_parser import SurveyInterface
    survey_interface = SurveyInterface(openai_api_key=openai_api_key, openai_model=openai_model, langsmith_api_key=langsmith_api_key)
    survey_contents = survey_interface.read_survey_contents(os.path.expanduser(input_path), use_llm=False)

2. In the second stage, the Markdown text is parsed into a structured format including modules, questions, response
   options, and so on. This is done by the ``parse_survey_contents()`` function, which uses an LLM to assist with
   parsing. For example::

    data = survey_interface.parse_survey_contents(survey_contents=survey_contents, survey_context=evaluation_context)

See the `ai_workflows <https://github.com/higherbar-ai/ai-workflows>`_ documentation for more details on how particular
file formats are read.

When parsing unstructured files into a structured survey format, a lot can go wrong. If your survey file is not being
read or parsed well, you might want to simplify the file to make it easier to read. For example:

1. Make sure that separate modules are in separate sections with clear headings.

2. Make sure that questions are clearly separated from one another, each with a unique identifier of some kind.

3. Make sure that response options are clearly separated from questions, and that they are clearly associated with the
   questions they belong to.

4. Label each translation with the same unique question identifier to help link them together. When possible, keep
   translations together.

After you've parsed a file, you can use the ``output_parsed_data_to_xlsform()`` method if you'd like to output it as an
XLSForm file formatted for SurveyCTO.

Known issues
^^^^^^^^^^^^

These known issues are inherited from `the ai_workflows package <https://github.com/higherbar-ai/ai-workflows>`_:

#. The example Google Colab notebooks pop up a message during installation that offers to restart the runtime. You have
   to click cancel so as not to interrupt execution.

#. The automatic generation and caching of JSON schemas (for response validation) can work poorly when batches of
   similar requests are all launched in parallel (as each request will generate and cache the schema).

#. When reading REDCap data dictionaries, translations aren't supported.

#. LangSmith tracing support is imperfect in a few ways:

   a. For OpenAI models, the top-level token usage counts are roughly doubled. You have to look to the inner LLM call
      for an accurate count of input and output tokens.
   b. For Anthropic models, the token usage doesn't show up at all, but you can find it by clicking into the metadata
      for the inner LLM call.
   c. For Anthropic models, the system prompt is only visible if you click into the inner LLM call and then switch the
      *Input* display to *Raw input*.
   d. For Anthropic models, images in prompts don't show properly.

Roadmap
-------

There's much that can be improved here. For example:

* We should track and report LLM costs.
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
#. Run the `initial-setup.ipynb <https://github.com/higherbar-ai/survey-eval/blob/main/src/initial-setup.ipynb>`_
   Jupyter notebook

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
