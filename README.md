# survey-eval

This repository contains a toolkit for AI-powered survey instrument evaluation. It's still in early development, but 
is ready to support piloting and experimentation. To learn more about the overall project, see 
[this blog post](https://www.linkedin.com/pulse/beating-mean-high-stakes-ai-christopher-robert-oodie/).  

Here are the basics:

1. This toolkit includes code to read, parse, and evaluate survey instruments.
2. The `file-evaluation-example.ipynb` Jupyter workbook provides a working example for evaluating a single survey
   instrument file. It includes details on how to install, configure, and run.
3. The evaluation engine itself lives in the `evaluation_engine.py` module. It provides a pretty basic framework for
   applying different evaluation lenses to a survey instrument.
4. The `core_evaluation_lenses.py` module contains an initial set of evaluation lenses that can be applied to survey 
   instruments. These are the ones applied in the example workbook. They are:
   1. `PhrasingEvaluationLens`: Cases where phrasing might be adjusted to improve respondent understanding and reduce 
      measurement error (i.e., the kinds of phrasing issues that would be identified through rigorous cognitive 
      interviewing or other forms of validation)
   2. `TranslationEvaluationLens`: Cases where translations are inaccurate or phrased such that they might lead to 
      differing response patterns
   3. `BiasEvaluationLens`: Cases where phrasing might be improved to remove implicit bias or stigmatizing language 
      (inspired by [this very helpful post](https://www.linkedin.com/pulse/using-chatgpt-counter-bias-prejudice-discrimination-johannes-schunter/) 
      on the subject of using ChatGPT to identify bias)
   4. `ValidatedInstrumentEvaluationLens`: Cases where a validated instrument might be adapted to better measure an 
      inferred construct of interest
5. The code for reading files is in the `questionnaire_file_reader.py` module. There's much there that can be improved 
   about how different file formats are read into raw text.
6. The code for parsing files is in the `questionnaire_file_parser.py` module. There's also a lot that can be done to
   improve how raw text is parsed into questions, modules, and so on. In particular, one might improve the range of
   examples provided to the LLM.

You can run the `file-evaluation-example.ipynb` workbook as-is, but you might also consider customizing the
core evaluation lenses to better meet your needs and/or adding your own evaluation lenses to the workbook. When adding
new lenses, you can just use any of the initial lenses as a template.

If you make use of this toolkit, we'd love to hear from you — and help to share your results with the community. Please
email us at `info@higherbar.ai`.

## Technical notes

### Reading input files

The `questionnaire_file_reader.py` module contains code for reading input files. It currently supports the following
file formats:

1. `.docx`: Word files are read in the `read_docx()` function, using LangChain's `UnstructuredFileLoader()` function.
   Note that text within "content controls" is effectively invisible, so you might need to open your file, select all, 
   and select "Remove content controls" to render the text visible 
   ([see here for more on content controls](https://learn.microsoft.com/en-us/office/client-developer/word/content-controls-in-word)).
2. `.pdf`: PDF files are read in the `read_pdf_combined()` function, which tries to read the text and tables in a PDF
   (separately), combine them together, and then fall back to using an OCR reader if that process didn't find much 
   text. There is a ton of room for improvement here.
3. `.xlsx`: Excel files are read in the `parse_xlsx()` function, in two stages. First, it assumes that the file is in
   [XLSForm format](https://xlsform.org/en/) and uses [the pyxform library](https://github.com/XLSForm/pyxform) to
   read the survey. If it encounters an error, it falls back to using LangChain's `UnstructuredExcelLoader()` to load
   the workbook in HTML format, then uses that HTML as the raw text for parsing. There is much that can be improved,
   particularly in how XLSForms are handled (e.g., the current approach doesn't handle translations well).
4. `.csv`: CSV files are read in the `parse_csv()` function, also using two stages. First, it assumes that the file
   is a REDCap data dictionary and parses the columns accordingly. If it encounters an error, it falls back to just
   reading the file as raw text. There is much that can be improved here, particularly in how REDCap data 
   dictionaries are handled (e.g., the current approach doesn't handle modules or translations).
5. `.html`: HTML files are read in the `read_html()` function, then converted into markdown for parsing.

All of the raw content is split into 7,500-character chunks with 500 characters of overlap, before being passed on
for parsing. This is necessary to both (a) avoid overflowing LLM context windows, and (b) allow the LLM to focus on
a tractable amount of text in any given request.

### Parsing input files

The actual parsing happens with LLM assistance, via [the kor library](https://github.com/eyurtsev/kor). All of that
code lives in `questionnaire_file_parser.py`, with the core parsing instructions and examples in `create_schema()`.

### Tracking and reporting costs

The API usage — including cost estimates — is currently output to the console as INFO-level logs, but only for the
parsing stage. The evaluation stage doesn't currently track or report costs.

## Roadmap

There's much that can be improved here. For example:

* We should track and report costs for the evaluation stage of the process.
* We should generally overhaul the `questionnaire_file_reader.py` module to better ingest different file formats into
  raw text that works consistently well for parsing. Better PDF, XLSForm, and REDCap support, in particular, would be
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
