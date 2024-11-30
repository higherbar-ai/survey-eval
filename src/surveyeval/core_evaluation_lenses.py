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

"""Core set of instrument evaluation lenses."""

from surveyeval import EvaluationEngine, EvaluationLens
from overrides import overrides
import json


class PhrasingEvaluationLens(EvaluationLens):
    """
    Lens for identifying phrasing issues that might be flagged during piloting or cognitive interviewing.
    """

    def __init__(self, evaluation_engine: EvaluationEngine):
        """
        Override default constructor to provide lens-specific prompt template and followup questions.

        :param evaluation_engine: EvaluationEngine instance to use for evaluation.
        :type evaluation_engine: EvaluationEngine
        """

        lens_eval_description = """This lens evaluates the phrasing of questions in a survey instrument, considering the context and locations provided. The goal is to identify any phrasing issues that might be flagged during piloting or cognitive interviewing. The lens will provide a list of phrases that are likely to be identified as problematic, along with suggested replacement phrases."""

        lens_system_prompt_template = """You are an AI designed to evaluate questionnaires and other survey instruments used by researchers and M&E professionals. You are an expert in survey methodology with training equivalent to a member of the American Association for Public Opinion Research (AAPOR) with a Ph.D. in survey methodology from University of Michigan’s Institute for Social Research. You consider primarily the content, context, and questions provided to you, and then content and methods from the most widely-cited academic publications and public and nonprofit research organizations.

You always give truthful, factual answers. When asked to give your response in a specific format, you always give your answer in the exact format requested. You never give offensive responses. If you don’t know the answer to a question, you truthfully say you don’t know.

You will be given an excerpt from a questionnaire or survey instrument between |!| and |!| delimiters. You will also be given a specific question from that excerpt to evaluate between |@| and |@| delimiters. Evaluate the question only, but also consider its context within the larger excerpt.

The broader context and location(s) for that excerpt are as follows. Consider these when evaluating the question.

Survey context: {survey_context}

Survey locations: {survey_locations}

Assume that this survey will be administered by a trained enumerator who asks each question and reads each prompt or instruction as indicated in the excerpt. Your job is to anticipate the phrasing or translation issues that would be identified in a rigorous process of pre-testing (with cognitive interviewing) and piloting.

When evaluating the question, DO:

1. Ensure that the question will be understandable by substantially all respondents, based on the survey context and locations above.

2. Consider the language, location, and survey context (with the appropriate location for the language coming from the "Survey locations" list above). 

3. Consider the question in the context of the excerpt, including any instructions, related questions, or prompts that precede it.

4. Ignore question numbers and formatting.

5. Assume that code to dynamically insert earlier responses or preloaded information like [FIELDNAME] or ${{{{fieldname}}}} is okay as it is.

6. Ignore HTML or other formatting, and focus solely on question phrasing (assume that HTML tags will be for visual formatting only and will not be read aloud).

When evaluating the question, DON'T: 

1. Recommend translating something into another language (i.e., suggestions for rephrasing should always be in the same language as the original text).

2. Recommend changes in the overall structure of a question (e.g., changing from multiple choice to open-ended or splitting one question into multiple), unless it will substantially improve the quality of the data collected.

3. Comment on HTML tags or formatting.

Respond in JSON format with all of the following fields:

* Phrases: a list containing all phrases from the excerpt that pre-testing or piloting is likely to identify as problematic (each phrase should be an exact quote)

* Number of phrases: the exact number of phrases in Phrases [ Note that this key must be exactly "Number of phrases", with exactly that capitalization and spacing ]

* Recommendations: a list containing suggested replacement phrases, one for each of the phrases in Phrases (in the same order as Phrases; each replacement phrase should be an exact quote that can exactly replace the corresponding phrase in Phrases; and each replacement phrase should be in the same language as the original phrase)

* Explanations: a list containing explanations for why the authors should consider revising each phrase, one for each of the phrases in Phrases (in the same order as Phrases). Do not repeat the entire phrase in the explanation, but feel free to reference specific words or parts as needed.

* Severities: a list containing the severity of each identified issue, one for each of the phrases in Phrases (in the same order as Phrases); each severity should be expressed as a number on a scale from 1 for the least severe issues (minor phrasing issues that are very unlikely to substantively affect responses) to 5 for the most severe issues (problems that are likely to substantively affect responses in a way that introduces bias and/or variance)
"""

        lens_question_template = """Excerpt: |!|{survey_excerpt}|!|\n\nQuestion: |@|{survey_question}|@|"""

        lens_followups = [
            {
                'condition_func': EvaluationLens.condition_list_has_less_than_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you certain that there are no phrasing issues likely to be identified in cognitive interviewing or piloting? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}."""
            },
            {
                'condition_func': EvaluationLens.condition_list_has_greater_or_equal_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you certain that each of the following statements are true of your earlier response?

1. The Phrases, Recommendations, Explanations, and Severities lists each have exactly {Number of phrases} elements, in the same parallel order.

2. Every element of the Severities list is a 1, 2, 3, 4, or 5, depending on the severity of the identified issue (with the most minor phrasing issues receiving a 1 and the most serious phrasing issues receiving a 5).

3. You didn't complain about or try to replace dynamic codes like [FIELDNAME] or ${{fieldname}}.

4. Your replacement phrase is in the same language as the original phrase.

5. Your recommendation concerns the specific question you were asked to evaluate.
 
6. You didn't complain about HTML tags or other formatting, but instead focused exclusively on question phrasing.

Consider whether you made any mistakes or neglected any instructions in your original response. If so, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being very careful in your work."""
            }
        ]

        # call super constructor
        super().__init__(lens_system_prompt_template, lens_question_template, lens_followups, evaluation_engine,
                         lens_eval_description)

    @overrides
    def evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                 survey_excerpt: str = "", survey_question: str = "", **kwargs) \
            -> dict:
        """
        Override default evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument (for context).
        :type survey_excerpt: str
        :param survey_question: Specific question to focus on.
        :type survey_question: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return super().evaluate(chat_history=chat_history, survey_context=survey_context,
                                survey_locations=survey_locations,
                                survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt),
                                survey_question=EvaluationEngine.clean_whitespace(survey_question), **kwargs)

    @overrides
    async def a_evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                         survey_excerpt: str = "", survey_question: str = "", **kwargs) \
            -> dict:
        """
        Override default a_evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param survey_question: Specific question to focus on.
        :type survey_question: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return await super().a_evaluate(chat_history=chat_history, survey_context=survey_context,
                                        survey_locations=survey_locations,
                                        survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt),
                                        survey_question=EvaluationEngine.clean_whitespace(survey_question),
                                        **kwargs)

    def format_result(self, result: dict | None = None, minimum_importance: int = 0) -> str:
        """
        Format the evaluation result as a human-readable string.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :param minimum_importance: Minimum importance score for filtering results (defaults to 0, which doesn't filter).
        :type minimum_importance: int
        :return: Formatted evaluation result.
        :rtype: str
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # return empty string if no result is available to format
        if result_to_format is None:
            return ""

        # format and return result
        try:
            formatted_result = ""
            sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                    key=lambda idx: result_to_format["Severities"][idx], reverse=True)
            for i in sorted_indices:
                severity = result_to_format['Severities'][i]
                # only report findings with severity greater than or equal to minimum_importance
                if severity >= minimum_importance:
                    formatted_result += f"Severity {severity} finding (out of 5):\n\n"
                    formatted_result += f"{result_to_format['Explanations'][i]}\n\n"
                    formatted_result += f"Recommend replacing: {result_to_format['Phrases'][i]}\n"
                    formatted_result += f"          With this: {result_to_format['Recommendations'][i]}\n\n"
        except Exception as e:
            # include exception in returned results, with raw JSON results
            formatted_result = (f"Error occurred formatting result: {str(e)}\n\n"
                                f"Raw JSON: {json.dumps(result_to_format, indent=4)}")
            self.evaluation_engine.logger.error(formatted_result)
        return formatted_result

    def standardize_result(self, result: dict | None = None) -> list[dict]:
        """
        Reorganize the evaluation result into a list of recommendations in a standardized format.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :return: List of recommendations, each of which is a dict with the following keys: importance (int 1-5),
            replacement_original (str), replacement_suggested (str), explanation (str).
        :rtype: list[dict]
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # organize and return result
        return_list = []
        try:
            # first make sure we have a result to organize
            if result_to_format is not None and result_to_format["Number of phrases"] > 0:
                # sort the results by severity
                sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                        key=lambda idx: result_to_format["Severities"][idx], reverse=True)
                # run through each result, appending to our return list
                for i in sorted_indices:
                    return_list.append({
                        'importance': result_to_format['Severities'][i],
                        'replacement_original': result_to_format['Phrases'][i],
                        'replacement_suggested': result_to_format['Recommendations'][i],
                        'explanation': result_to_format['Explanations'][i]
                    })
        except Exception as e:
            raise RuntimeError(f"Error occurred formatting result: {str(e)}\n\n"
                               f"Raw JSON: {json.dumps(result_to_format, indent=4)}") from e
        return return_list


class ValidatedInstrumentEvaluationLens(EvaluationLens):
    """
    Lens for identifying validated questions, instruments, or tools that either were used or could be used to measure
    what the excerpt is attempting to measure.
    """

    def __init__(self, evaluation_engine: EvaluationEngine):
        """
        Override default constructor to provide lens-specific prompt template and followup questions.

        :param evaluation_engine: EvaluationEngine instance to use for evaluation.
        :type evaluation_engine: EvaluationEngine
        """

        lens_eval_description = """This lens evaluates an excerpt of questions from a survey instrument, considering the context and locations provided. The goal is to identify validated questions, instruments, or tools commonly used for measuring what the excerpt is attempting to measure, and recommend that the authors consider replicating or adapting these as appropriate."""

        lens_system_prompt_template = """You are an AI designed to evaluate questionnaires and other survey instruments used by researchers and M&E professionals. You are an expert in survey methodology with training equivalent to a member of the American Association for Public Opinion Research (AAPOR) with a Ph.D. in survey methodology from University of Michigan’s Institute for Social Research. You consider primarily the content, context, and questions provided to you, and then content and methods from the most widely-cited academic publications and public and nonprofit research organizations.

You always give truthful, factual answers. When asked to give your response in a specific format, you always give your answer in the exact format requested. You never give offensive responses. If you don’t know the answer to a question, you truthfully say you don’t know.

You will be given an excerpt from a questionnaire or survey instrument between |@| and |@| delimiters. The context and location(s) for that excerpt are as follows:

Survey context: {survey_context}

Survey locations: {survey_locations}

Your job is to respond with your evaluation in JSON format with all of the following fields:

* Measuring: a list containing one or more short strings describing what the excerpt seems to be attempting to measure

* Replication: 1 if the excerpt includes an exact replication of a validated question, instrument, or tool commonly used for measuring what the excerpt is attempting to measure; otherwise 0. Only answer 1 if the excerpt includes all of the same questions and response options as the validated version.

* Replication name: if Replication is 1, the name of the validated question, instrument, or tool that has been replicated; otherwise an empty string

* Replication URL: if Replication is 1, a URL to learn more about the validated question, instrument, or tool that has been replicated; otherwise an empty string. Only give a URL if you are confident that the URL is the right place to go to learn more, otherwise describe where to go to learn more.

* Replication explanation: if Replication is 1, a short description of how the excerpt includes the validated version; otherwise an empty string

* Adaptation: 1 if the excerpt includes an adapted version of a validated question, instrument, or tool commonly used for measuring what the excerpt is attempting to measure; otherwise 0. Only answer 1 if the excerpt includes strong similarity to the validated version, but not exactly the same questions or response options. If the excerpt includes only weak or superficial similarity to the validated version, answer 0.

* Adaptation name: if Adaptation is 1, the name of the validated question, instrument, or tool that has been adapted; otherwise an empty string

* Adaptation URL: if Adaptation is 1, a URL to learn more about the validated question, instrument, or tool that has been adapted; otherwise an empty string. Only give a URL if you are confident that the URL is the right place to go to learn more, otherwise describe where to go to learn more.

* Adaptation explanation: if Adaptation is 1, a short description of how the excerpt is similar to and different from the validated version; otherwise an empty string

* Recommendation: 1 if you would recommend that the author consider using a validated question, instrument, or tool commonly used for measuring what the excerpt is attempting to measure (or a different one, if Replication or Adaptation is 1), rather than the current approach; otherwise 0

* Recommendation name: if Recommendation is 1, the name of the validated question, instrument, or tool you would recommend; otherwise an empty string

* Recommendation URL: if Recommendation is 1, a URL to learn more about the validated question, instrument, or tool you would recommend; otherwise an empty string. Only give a URL if you are confident that the URL is the right place to go to learn more, otherwise describe where to go to learn more.

* Recommendation explanation: if Recommendation is 1, a short description of why the author should consider the validated version proposed; otherwise an empty string

* Recommendation strength: if Recommendation is 1, a number from 1 to 5 to indicate the strength of the recommendation, with 1 being the weakest possible recommendation and 5 being the strongest possible recommendation; otherwise an empty string. In deciding on a Recommendation strength, consider both what is being recommended and how good a fit it might be to the survey context and locations."""

        lens_question_template = """Excerpt: |@|{survey_excerpt}|@|"""

        lens_followups = [
            {
                'condition_func': EvaluationLens.condition_is_value,
                'condition_key': 'Replication',
                'condition_value': 1,
                'prompt_template': """Are you certain that the excerpt contains an exact replication of {Replication name}, including the same question(s) and answer option(s)? If the version in the excerpt is different in some way, then Replication should be 0 and Adaptation should be 1. If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_is_value,
                'condition_key': 'Adaptation',
                'condition_value': 1,
                'prompt_template': """Are you certain that the excerpt contains an adaptation of {Adaptation name}, meaning that it exhibits strong similarity to the validated version, but not exactly the same questions or response options? If the excerpt includes only weak or superficial similarity to the validated version, it should not be considered an adaptation. And if its questions and response options are exactly the same as a validated version, then it should be considered a Replication instead. If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_is_not_value,
                'condition_key': 'Replication URL',
                'condition_value': "",
                'prompt_template': """Are you certain that the Replication URL you supplied is the best place for the author to learn more? If not, please respond with a revised JSON response with a Replication URL that includes a different URL or a short description of where the author should go to learn more (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_is_not_value,
                'condition_key': 'Adaptation URL',
                'condition_value': "",
                'prompt_template': """Are you certain that the Adaptation URL you supplied is the best place for the author to learn more? If not, please respond with a revised JSON response with a Adaptation URL that includes a different URL or a short description of where the author should go to learn more (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_is_not_value,
                'condition_key': 'Recommendation URL',
                'condition_value': "",
                'prompt_template': """Are you certain that the Recommendation URL you supplied is the best place for the author to learn more? If not, please respond with a revised JSON response with a Recommendation URL that includes a different URL or a short description of where the author should go to learn more (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_is_value,
                'condition_key': 'Recommendation',
                'condition_value': 1,
                'prompt_template': """Are you certain that the Recommendation explanation and Recommendation strength you supplied is correct, and that the two are consistent with one another? If not, please respond with a revised JSON response that revises the recommendation details as appropriate (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            }
        ]

        # call super constructor
        super().__init__(lens_system_prompt_template, lens_question_template, lens_followups, evaluation_engine,
                         lens_eval_description)

    @overrides
    def evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                 survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return super().evaluate(chat_history=chat_history, survey_context=survey_context,
                                survey_locations=survey_locations,
                                survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    @overrides
    async def a_evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                         survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default a_evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return await super().a_evaluate(chat_history=chat_history, survey_context=survey_context,
                                        survey_locations=survey_locations,
                                        survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    def format_result(self, result: dict | None = None, minimum_importance: int = 0) -> str:
        """
        Format the evaluation result as a human-readable string.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :param minimum_importance: Minimum importance score for filtering results (defaults to 0, which doesn't filter).
        :type minimum_importance: int
        :return: Formatted evaluation result.
        :rtype: str
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # return empty string if no result is available to format
        if result_to_format is None:
            return ""

        # format and return result
        try:
            if minimum_importance > 0 and (not result_to_format["Recommendation"]
                                           or result_to_format["Recommendation strength"] < minimum_importance):
                # filter this result since as not important enough
                return ""

            if len(result_to_format["Measuring"]) == 0:
                measuring_str = ""
            elif len(result_to_format["Measuring"]) == 1:
                measuring_str = str(result_to_format["Measuring"][0])
            else:
                measuring_str = ', '.join(result_to_format["Measuring"])
            formatted_result = f"FYI: Excerpt likely attempting to measure {measuring_str}\n\n"

            if result_to_format["Replication"]:
                formatted_result = (f'FYI: Excerpt likely replication of "{result_to_format["Replication name"]}". '
                                    f'{result_to_format["Replication explanation"]} (Learn more: '
                                    f'{result_to_format["Replication URL"]})\n\n')
            if result_to_format["Adaptation"]:
                formatted_result = (f'FYI: Excerpt likely adaptation of "{result_to_format["Adaptation name"]}". '
                                    f'{result_to_format["Adaptation explanation"]} (Learn more: '
                                    f'{result_to_format["Adaptation URL"]})\n\n')

            if result_to_format["Recommendation"]:
                formatted_result = (f'Recommendation (strength {result_to_format["Recommendation strength"]} '
                                    f'out of 5):\n\n'
                                    f'Consider adapting: {result_to_format["Recommendation name"]}\n'
                                    f'Explanation: {result_to_format["Recommendation explanation"]}\n'
                                    f'Learn more: {result_to_format["Recommendation URL"]}\n\n')
        except Exception as e:
            # include exception in returned results, with raw JSON results
            formatted_result = (f"Error occurred formatting result: {str(e)}\n\n"
                                f"Raw JSON: {json.dumps(result_to_format, indent=4)}")
            self.evaluation_engine.logger.error(formatted_result)
        return formatted_result

    def standardize_result(self, result: dict | None = None) -> list[dict]:
        """
        Reorganize the evaluation result into a list of recommendations in a standardized format.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :return: List of recommendations, each of which is a dict with the following keys: importance (int 1-5),
            replacement_original (str), replacement_suggested (str), explanation (str).
        :rtype: list[dict]
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # organize and return result
        return_list = []
        try:
            # first make sure we have a meaningful result to organize
            if result_to_format is not None and result_to_format["Recommendation"]:
                # start with the recommendation itself
                explanation = (f"Consider adapting {result_to_format['Recommendation name']}. "
                               f"{result_to_format['Recommendation explanation']} "
                               f"(Learn more: {result_to_format['Recommendation URL']})")

                # add information about what the excerpt is attempting to measure (if any)
                if len(result_to_format["Measuring"]) == 0:
                    measuring_str = ""
                elif len(result_to_format["Measuring"]) == 1:
                    measuring_str = str(result_to_format["Measuring"][0])
                else:
                    measuring_str = ', '.join(result_to_format["Measuring"])
                if measuring_str:
                    explanation += f"\n\nFYI, you appear to be measuring {measuring_str}."

                if result_to_format["Replication"]:
                    explanation += (f'\n\nFYI, this could be a replication of '
                                    f'"{result_to_format["Replication name"]}". '
                                    f'{result_to_format["Replication explanation"]} (Learn more: '
                                    f'{result_to_format["Replication URL"]})')
                if result_to_format["Adaptation"]:
                    explanation += (f'\n\nFYI, this could be an adaptation of '
                                    f'"{result_to_format["Adaptation name"]}". '
                                    f'{result_to_format["Adaptation explanation"]} (Learn more: '
                                    f'{result_to_format["Adaptation URL"]})')

                # append our one and only recommendation to the return list
                return_list.append({
                    'importance': result_to_format['Recommendation strength'],
                    'replacement_original': "",
                    'replacement_suggested': "",
                    'explanation': explanation
                })
        except Exception as e:
            raise RuntimeError(f"Error occurred formatting result: {str(e)}\n\n"
                               f"Raw JSON: {json.dumps(result_to_format, indent=4)}") from e
        return return_list


class TranslationEvaluationLens(EvaluationLens):
    """
    Lens for identifying translation issues that could lead to differing response patterns from respondents.
    """

    def __init__(self, evaluation_engine: EvaluationEngine):
        """
        Override default constructor to provide lens-specific prompt template and followup questions.

        :param evaluation_engine: EvaluationEngine instance to use for evaluation.
        :type evaluation_engine: EvaluationEngine
        """

        lens_eval_description = """This lens evaluates the translation of questions in a survey instrument, considering the context and locations provided. The goal is to review the excerpt for differences in translation that could lead to differing response patterns from respondents. The lens will identify translations that are problematic and suggest replacements that make the translations accurate enough that data collected will be comparable regardless of the language of administration."""

        lens_system_prompt_template = """You are an AI designed to evaluate questionnaires and other survey instruments used by researchers and M&E professionals. You are an expert in survey methodology with training equivalent to a member of the American Association for Public Opinion Research (AAPOR) with a Ph.D. in survey methodology from University of Michigan’s Institute for Social Research. You consider primarily the content, context, and questions provided to you, and then content and methods from the most widely-cited academic publications and public and nonprofit research organizations.

You always give truthful, factual answers. When asked to give your response in a specific format, you always give your answer in the exact format requested. You never give offensive responses. If you don’t know the answer to a question, you truthfully say you don’t know.

You will be given an excerpt from a questionnaire or survey instrument between |@| and |@| delimiters. The context and location(s) for that excerpt are as follows:

Survey context: {survey_context}

Survey locations: {survey_locations}

The excerpt will include the same questions and response options in two languages, one labeled as primary and one labeled as translated. Assume that this survey will be administered by a trained enumerator who asks each question in a single language appropriate to the respondent and reads each prompt or instruction as indicated in the excerpt. Your job is to review the excerpt for differences in the translation that could lead to differing response patterns from respondents. The goal is for translations to be accurate enough that data collected will be comparable regardless of the language of administration.

Assume that each language will be used in the location contexts as indicated in the "Survey locations" details above, and consider those location contexts (and only those location contexts) in your evaluation. For example, when evaluating a question in French or English, consider the appropriate country contexts listed in the "Survey locations", as languages might have different dialects and conventions in different countries.

Also assume that translations should be designed for understandability and comparability across locations. Literal translations might not best achieve the same meaning and response patterns across settings. Ensure that the meaning is common across languages and locations.

Finally, identify problematic phrases and replacement text in the translated language only. Do not propose any changes in the primary language.

Respond in JSON format with all of the following fields:

* Phrases: a list containing all problematic phrases from the translation that you found in your review, where the translation does not adequately match the meaning in the primary language (each phrase should be an exact quote from the translated language)

* Number of phrases: the exact number of phrases in Phrases [ Note that this key must be exactly "Number of phrases", with exactly that capitalization and spacing ]

* Recommendations: a list containing suggested replacement phrases, one for each of the phrases in Phrases (in the same order as Phrases; each replacement phrase should be an exact quote that can exactly replace the corresponding phrase in Phrases; and each replacement phrase should be in the same translated language as the phrase it replaces)

* Explanations: a list containing explanations for why the phrases are problematic, one for each of the phrases in Phrases (in the same order as Phrases)

* Severities: a list containing the severity of each identified issue, one for each of the phrases in Phrases (in the same order as Phrases); each severity should be expressed as a number on a scale from 1 for the least severe issues (minor phrasing issues that are very unlikely to substantively affect response patterns) to 5 for the most severe issues (problems that are very likely to substantively affect response patterns in a way that introduces bias and/or variance)"""

        lens_question_template = """Excerpt: |@|{survey_excerpt}|@|"""

        lens_followups = [
            {
                'condition_func': EvaluationLens.condition_list_has_less_than_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you sure that there are no inaccurate translations, nor any response options that are missing or ordered differently in the translated version? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_list_has_greater_or_equal_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Differences in the meaning of questions or response options can affect response patterns in important ways. Are you sure that you didn't miss any cases where differences in meaning could lead to different response patterns? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_list_has_greater_or_equal_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """If the same response options are not present in all translations in the same order, response patterns can differ. Are you sure that you didn't miss any cases where response options are missing from the translation, or in a different order? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_list_has_greater_or_equal_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you certain that (1) the Phrases, Recommendations, Explanations, and Severities lists each have exactly {Number of phrases} elements, in the same parallel order; (2) every element of the Severities list is a 1, 2, 3, 4, or 5, depending on the severity of the issue; and (3) every phrase in Phrases and every recommendation in Recommendations is in the translated language? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            }
        ]

        # call super constructor
        super().__init__(lens_system_prompt_template, lens_question_template, lens_followups, evaluation_engine,
                         lens_eval_description)

    @overrides
    def evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                 survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return super().evaluate(chat_history=chat_history, survey_context=survey_context,
                                survey_locations=survey_locations,
                                survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    @overrides
    async def a_evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                         survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default a_evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return await super().a_evaluate(chat_history=chat_history, survey_context=survey_context,
                                        survey_locations=survey_locations,
                                        survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    def format_result(self, result: dict | None = None, minimum_importance: int = 0) -> str:
        """
        Format the evaluation result as a human-readable string.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :param minimum_importance: Minimum importance score for filtering results (defaults to 0, which doesn't filter).
        :type minimum_importance: int
        :return: Formatted evaluation result.
        :rtype: str
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # return empty string if no result is available to format
        if result_to_format is None:
            return ""

        # format and return result
        try:
            formatted_result = ""
            sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                    key=lambda idx: result_to_format["Severities"][idx], reverse=True)
            for i in sorted_indices:
                severity = result_to_format['Severities'][i]
                # only report findings with severity greater than or equal to minimum_importance
                if severity >= minimum_importance:
                    formatted_result += f"Severity {severity} finding (out of 5):\n\n"
                    formatted_result += f"{result_to_format['Explanations'][i]}\n\n"
                    formatted_result += f"Recommend replacing: {result_to_format['Phrases'][i]}\n"
                    formatted_result += f"          With this: {result_to_format['Recommendations'][i]}\n\n"
        except Exception as e:
            # include exception in returned results, with raw JSON results
            formatted_result = (f"Error occurred formatting result: {str(e)}\n\n"
                                f"Raw JSON: {json.dumps(result_to_format, indent=4)}")
            self.evaluation_engine.logger.error(formatted_result)
        return formatted_result

    def standardize_result(self, result: dict | None = None) -> list[dict]:
        """
        Reorganize the evaluation result into a list of recommendations in a standardized format.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :return: List of recommendations, each of which is a dict with the following keys: importance (int 1-5),
            replacement_original (str), replacement_suggested (str), explanation (str).
        :rtype: list[dict]
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # organize and return result
        return_list = []
        try:
            # first make sure we have a result to organize
            if result_to_format is not None and result_to_format["Number of phrases"] > 0:
                # sort the results by severity
                sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                        key=lambda idx: result_to_format["Severities"][idx], reverse=True)
                # run through each result, appending to our return list
                for i in sorted_indices:
                    return_list.append({
                        'importance': result_to_format['Severities'][i],
                        'replacement_original': result_to_format['Phrases'][i],
                        'replacement_suggested': result_to_format['Recommendations'][i],
                        'explanation': result_to_format['Explanations'][i]
                    })
        except Exception as e:
            raise RuntimeError(f"Error occurred formatting result: {str(e)}\n\n"
                               f"Raw JSON: {json.dumps(result_to_format, indent=4)}") from e
        return return_list


class BiasEvaluationLens(EvaluationLens):
    """
    Lens for evaluating a survey excerpt for any of the following:

    * Stereotypical representations of gender, ethnicity, origin, religion, or other social categories.
    * Distorted or biased representations of events, topics, groups, or individuals.
    * Use of discriminatory or insensitive language towards certain groups or topics.
    * Implicit or explicit assumptions made in the text or unquestioningly adopted that could be based on prejudices.
    * Prejudiced descriptions or evaluations of abilities, characteristics, or behaviors.
    """

    def __init__(self, evaluation_engine: EvaluationEngine):
        """
        Override default constructor to provide lens-specific prompt template and followup questions.

        :param evaluation_engine: EvaluationEngine instance to use for evaluation.
        :type evaluation_engine: EvaluationEngine
        """

        lens_eval_description = """This lens evaluates the phrasing of questions in a survey instrument, considering the context and locations provided. The goal is to identify phrasing that includes stereotypes, bias, discriminatory or insensitive language, or prejudice. The lens will provide a list of phrases that are likely to be identified as problematic, along with suggested replacement phrases."""

        # Note that this prompt was constructed from the example in this blog post:
        # https://www.linkedin.com/pulse/using-chatgpt-counter-bias-prejudice-discrimination-johannes-schunter/
        lens_system_prompt_template = """You are an AI designed to evaluate questionnaires and other survey instruments used by researchers and M&E professionals. You are an expert in survey methodology with training equivalent to a member of the American Association for Public Opinion Research (AAPOR) with a Ph.D. in survey methodology from University of Michigan’s Institute for Social Research. You are also an expert in the areas of gender equality, discrimination, anti-racism, and anti-colonialism. You consider primarily the content, context, and questions provided to you, and then content and methods from the most widely-cited academic publications and public and nonprofit research organizations.

You always give truthful, factual answers. When asked to give your response in a specific format, you always give your answer in the exact format requested. You never give offensive responses. If you don’t know the answer to a question, you truthfully say you don’t know.

You will be given an excerpt from a questionnaire or survey instrument between |@| and |@| delimiters. The context and location(s) for that excerpt are as follows:

Survey context: {survey_context}

Survey locations: {survey_locations}

Assume that this survey will be administered by a trained enumerator who asks each question and reads each prompt or instruction as indicated in the excerpt. Your job is to review the excerpt for:

a. Stereotypical representations of gender, ethnicity, origin, religion, or other social categories.

b. Distorted or biased representations of events, topics, groups, or individuals.

c. Use of discriminatory or insensitive language towards certain groups or topics.

d. Implicit or explicit assumptions made in the text or unquestioningly adopted that could be based on prejudices.

e. Prejudiced descriptions or evaluations of abilities, characteristics, or behaviors.

Respond in JSON format with all of the following fields:

* Phrases: a list containing all problematic phrases from the excerpt that you found in your review (each phrase should be an exact quote from the excerpt)

* Number of phrases: the exact number of phrases in Phrases [ Note that this key must be exactly "Number of phrases", with exactly that capitalization and spacing ]

* Recommendations: a list containing suggested replacement phrases, one for each of the phrases in Phrases (in the same order as Phrases; each replacement phrase should be an exact quote that can exactly replace the corresponding phrase in Phrases)

* Explanations: a list containing explanations for why the phrases are problematic, one for each of the phrases in Phrases (in the same order as Phrases)

* Severities: a list containing the severity of each identified issue, one for each of the phrases in Phrases (in the same order as Phrases); each severity should be expressed as a number on a scale from 1 for the least severe issues (minor phrasing issues that are very unlikely to offend respondents or substantively affect their responses) to 5 for the most severe issues (problems that are very likely to offend respondents or substantively affect responses in a way that introduces bias and/or variance)"""

        lens_question_template = """Excerpt: |@|{survey_excerpt}|@|"""

        lens_followups = [
            {
                'condition_func': EvaluationLens.condition_list_has_less_than_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you certain that there are no problematic phrases? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            },
            {
                'condition_func': EvaluationLens.condition_list_has_greater_or_equal_elements,
                'condition_key': 'Phrases',
                'condition_value': 1,
                'prompt_template': """Are you certain that (1) the Phrases, Recommendations, Explanations, and Severities lists each have exactly {Number of phrases} elements, in the same parallel order; and (2) every element of the Severities list is a 1, 2, 3, 4, or 5, depending on the severity of the issue? If appropriate, please respond with a revised JSON response (including all fields). If you have no changes to propose, respond with an empty JSON response of {{}}. Thank you for being careful in your work."""
            }
        ]

        # call super constructor
        super().__init__(lens_system_prompt_template, lens_question_template, lens_followups, evaluation_engine,
                         lens_eval_description)

    @overrides
    def evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                 survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return super().evaluate(chat_history=chat_history, survey_context=survey_context,
                                survey_locations=survey_locations,
                                survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    @overrides
    async def a_evaluate(self, chat_history: list = None, survey_context: str = "", survey_locations: str = "",
                         survey_excerpt: str = "", **kwargs) -> dict:
        """
        Override default a_evaluate method.

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param survey_context: Information about the survey context.
        :type survey_context: str
        :param survey_locations: Information about the survey location(s).
        :type survey_locations: str
        :param survey_excerpt: Excerpt from the survey instrument to evaluate.
        :type survey_excerpt: str
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        return await super().a_evaluate(chat_history=chat_history, survey_context=survey_context,
                                        survey_locations=survey_locations,
                                        survey_excerpt=EvaluationEngine.clean_whitespace(survey_excerpt), **kwargs)

    def format_result(self, result: dict | None = None, minimum_importance: int = 0) -> str:
        """
        Format the evaluation result as a human-readable string.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :param minimum_importance: Minimum importance score for filtering results (defaults to 0, which doesn't filter).
        :type minimum_importance: int
        :return: Formatted evaluation result.
        :rtype: str
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # return empty string if no result is available to format
        if result_to_format is None:
            return ""

        # format and return result
        try:
            formatted_result = ""
            sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                    key=lambda idx: result_to_format["Severities"][idx], reverse=True)
            for i in sorted_indices:
                severity = result_to_format['Severities'][i]
                # only report findings with severity greater than or equal to minimum_importance
                if severity >= minimum_importance:
                    formatted_result += f"Severity {severity} finding (out of 5):\n\n"
                    formatted_result += f"{result_to_format['Explanations'][i]}\n\n"
                    formatted_result += f"Recommend replacing: {result_to_format['Phrases'][i]}\n"
                    formatted_result += f"          With this: {result_to_format['Recommendations'][i]}\n\n"
        except Exception as e:
            # include exception in returned results, with raw JSON results
            formatted_result = (f"Error occurred formatting result: {str(e)}\n\n"
                                f"Raw JSON: {json.dumps(result_to_format, indent=4)}")
            self.evaluation_engine.logger.error(formatted_result)
        return formatted_result

    def standardize_result(self, result: dict | None = None) -> list[dict]:
        """
        Reorganize the evaluation result into a list of recommendations in a standardized format.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :return: List of recommendations, each of which is a dict with the following keys: importance (int 1-5),
            replacement_original (str), replacement_suggested (str), explanation (str).
        :rtype: list[dict]
        """

        # use evaluation_result attribute if no result is passed
        if result is not None:
            result_to_format = result
        else:
            result_to_format = self.evaluation_result

        # organize and return result
        return_list = []
        try:
            # first make sure we have a result to organize
            if result_to_format is not None and result_to_format["Number of phrases"] > 0:
                # sort the results by severity
                sorted_indices = sorted(range(len(result_to_format["Severities"])),
                                        key=lambda idx: result_to_format["Severities"][idx], reverse=True)
                # run through each result, appending to our return list
                for i in sorted_indices:
                    return_list.append({
                        'importance': result_to_format['Severities'][i],
                        'replacement_original': result_to_format['Phrases'][i],
                        'replacement_suggested': result_to_format['Recommendations'][i],
                        'explanation': result_to_format['Explanations'][i]
                    })
        except Exception as e:
            raise RuntimeError(f"Error occurred formatting result: {str(e)}\n\n"
                               f"Raw JSON: {json.dumps(result_to_format, indent=4)}") from e
        return return_list
