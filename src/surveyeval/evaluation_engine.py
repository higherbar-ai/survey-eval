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

"""Core classes for instrument evaluation engine."""

from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables import Runnable
from langchain_core.runnables.history import RunnableWithMessageHistory
import re
import tiktoken
import asyncio
import logging
import copy


class EvaluationEngine:
    """Main class for instrument evaluation engine."""

    summarize_model: str
    summarize_provider: str
    evaluation_model: str
    evaluation_provider: str
    openai_api_key: str
    azure_api_key: str
    azure_api_base: str
    azure_api_version: str
    tiktoken_model_name: str
    tokenizer: tiktoken.Encoding
    temperature: float
    max_retries: int
    logger: logging.Logger
    extra_evaluation_instructions: str

    def __init__(self, summarize_model: str, summarize_provider: str, evaluation_model: str, evaluation_provider: str,
                 openai_api_key: str, azure_api_key: str = "", azure_api_base: str = "", azure_api_version: str = "",
                 temperature: float = 0.1, tiktoken_model_name: str = "", max_retries: int = 3,
                 logger: logging.Logger = None, extra_evaluation_instructions: str = ""):
        """
        Initialize evaluation engine.

        :param summarize_model: LLM model to use for summarizing multistep conversations (not currently used).
        :type summarize_model: str
        :param summarize_provider: Provider name for the summarization model ("openai" or "azure") (not currently used).
        :type summarize_provider: str
        :param evaluation_model: LLM model to use for instrument evaluation (when using Azure, the
            deployment or engine name must be the same as the model name).
        :type evaluation_model: str
        :param evaluation_provider: Provider name for the evaluation model ("openai" or "azure").
        :type evaluation_provider: str
        :param openai_api_key: API key for OpenAI services (if evaluation_provider is "openai").
        :type openai_api_key: str
        :param azure_api_key: API key for Azure services (if evaluation_provider is "azure").
        :type azure_api_key: str
        :param azure_api_base: Base URL for Azure API (if evaluation_provider is "azure").
        :type azure_api_base: str
        :param azure_api_version: Version of the Azure API (if evaluation_provider is "azure").
        :type azure_api_version: str
        :param temperature: Temperature setting for AI model responses.
        :type temperature: float
        :param tiktoken_model_name: Name of the model used with TikToken, if different from evaluation_model.
        :type tiktoken_model_name: str
        :param logger: Logger instance for logging messages.
        :type logger: logging.Logger
        :param extra_evaluation_instructions: Extra evaluation instructions (if any).
        :type extra_evaluation_instructions: str
        """

        # initialize object from constructor parameters
        self.summarize_model = summarize_model
        self.summarize_provider = summarize_provider
        self.evaluation_model = evaluation_model
        self.evaluation_provider = evaluation_provider
        self.openai_api_key = openai_api_key
        self.azure_api_key = azure_api_key
        self.azure_api_base = azure_api_base
        self.azure_api_version = azure_api_version
        self.temperature = temperature
        if not tiktoken_model_name:
            self.tiktoken_model_name = evaluation_model
        else:
            self.tiktoken_model_name = tiktoken_model_name
        self.tokenizer = tiktoken.encoding_for_model(self.tiktoken_model_name)
        self.max_retries = max_retries
        if not logger:
            self.logger = logging.getLogger(__name__)
        else:
            self.logger = logger
        self.extra_evaluation_instructions = extra_evaluation_instructions

    def tiktoken_len(self, text: str) -> int:
        """
        Count the number of tokens in a text string.

        :param text: Text to count tokens in.
        :type text: str
        :return: Count of tokens in text.
        :rtype: int
        """

        # convert to tokens and return count
        tokens = self.tokenizer.encode(
            text,
            disallowed_special=()
        )
        return len(tokens)

    def trim_string(self, s: str, max_tokens: int) -> str:
        """
        Trim string as needed to stay within token limit.

        :param s: String to trim.
        :type s: str
        :param max_tokens: Maximum number of tokens to allow in string.
        :type max_tokens: int
        :return: Trimmed string.
        :rtype: str
        """

        # see if the string is over the limit
        tokens = list(self.tokenizer.encode(s))
        if len(tokens) > max_tokens:
            # if so, truncate the tokens list and convert it back to a string
            tokens = tokens[:max_tokens]
            s = self.tokenizer.decode(tokens)

        return s

    @staticmethod
    def trim_json(json_str: str) -> str:
        """
        Trim common leading and trailing characters from JSON string.

        :param json_str: JSON string to trim
        :type json_str: str
        :return: Trimmed JSON string
        :rtype: str
        """

        # trim leading and trailing whitespace
        trimmed_json = json_str.strip()
        # trim enclosing code blocks
        trimmed_json = re.sub(r'^```json\n', '', trimmed_json)
        trimmed_json = re.sub(r'^```\n', '', trimmed_json)
        trimmed_json = re.sub(r'\n```$', '', trimmed_json)

        return trimmed_json

    @staticmethod
    def clean_whitespace(s: str) -> str:
        """
        Strip whitespace from prompt string in order to economize on tokens.

        :param s: Prompt string to clean.
        :type s: str
        :return: Cleaned prompt string.
        :rtype: str
        """

        # convert tabs to spaces
        s = s.replace('\t', ' ')

        # convert multiple spaces to one
        s = re.sub(' +', ' ', s)

        # remove the space from lines that only include a space
        s = re.sub('\n \n', '\n\n', s)
        s = re.sub('^ $', '', s, flags=re.MULTILINE)

        return s

    def get_chain(self, system_prompt: str = "", starting_chat_history: list[tuple] = None) -> Runnable:
        """
        Get a conversation chain for use in evaluating an instrument.

        :param system_prompt: System prompt template to use for the conversation chain.
        :type system_prompt: str
        :param starting_chat_history: Starting chat history to use for the conversation chain (or None for none).
        :type starting_chat_history: list[tuple]
        :return: Runnable conversation chain to use for instrument evaluation.
        :rtype: Runnable
        """

        # initialize evaluation provider
        if self.evaluation_provider == "azure":
            # for this Azure implementation, the engine name always has to match the model
            conversation_llm = AzureChatOpenAI(
                temperature=self.temperature,
                verbose=False,
                model_name=self.evaluation_model,
                azure_endpoint=self.azure_api_base,
                openai_api_version=self.azure_api_version,
                deployment_name=self.evaluation_model,
                openai_api_key=self.azure_api_key,
                openai_api_type="azure",
                tiktoken_model_name=self.tiktoken_model_name,
                model=self.evaluation_model
            )
        else:
            conversation_llm = ChatOpenAI(
                verbose=False,
                temperature=self.temperature,
                model_name=self.evaluation_model,
                openai_api_key=self.openai_api_key,
                tiktoken_model_name=self.tiktoken_model_name
            )

        # initialize chat history
        history = InMemoryChatMessageHistory()
        if starting_chat_history is not None:
            for (human_message, ai_message) in starting_chat_history:
                history.add_user_message(human_message)
                history.add_ai_message(ai_message)

        # create template for chat prompt, including history
        chat_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("placeholder", "{chat_history}"),
                ("human", "{question}"),
            ]
        )

        # create and return a runnable chain
        chain = chat_prompt | conversation_llm.with_structured_output(method="json_mode", include_raw=True)
        chain_with_history = RunnableWithMessageHistory(chain, lambda x: history, input_messages_key="question",
                                                        history_messages_key="chat_history",
                                                        output_messages_key="raw")
        return chain_with_history

    async def a_run_evaluation_chain(self, task_system_prompt: str, question: str, followups: list[dict],
                                     chat_history: list = None) -> dict:
        """
        Run an evaluation chain (asynchronously).

        :param task_system_prompt: System prompt template to use for the evaluation chain. This can include the
            {survey_context} and {survey_locations} variables to include information about the survey context. It
            should specify a specific JSON format for all responses.
        :type task_system_prompt: str
        :param question: Initial question to ask, to begin the evaluation chain. This should include the
            {survey_excerpt} variable to include the appropriate excerpt being evaluated.
        :type question: str
        :param followups: List of follow-up questions to ask, based on the JSON response to the initial question.
            Should be a list of dicts, each with a value for each of the following keys:
            "condition_func": function that returns True when the follow-up should be asked and False when it
            shouldn't (should take the following parameters: response_dict: dict, condition_key: str,
            condition_value: value to check against the one in the response dict);
            "condition_key": the key in the response dict to check;
            "condition_value": the value to check against the one in the response dict;
            "prompt_template": the template for the follow-up question to ask (which can include variables from
            the response dict).
        :type followups: list[dict]
        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        # initialize new evaluation chain
        llm_chain = self.get_chain(system_prompt=task_system_prompt, starting_chat_history=chat_history)

        # initialize results
        result_dict = {"result": None, "error": None, "response": None, "history": None}
        full_history = []

        # ask our question to the LLM, retrying the appropriate number of times
        attempt = 0
        while True:
            attempt += 1
            try:
                # ask our question and break on success
                result = await llm_chain.ainvoke({"question": question},
                                                 config={"configurable": {"session_id": "NA"}})
                if 'parsed' in result and result['parsed'] is not None:
                    # if we got a parsed result, break from retry loop
                    break
                elif 'parsing_error' in result and result['parsing_error'] is not None:
                    # if there was a parsing error, raise an exception
                    raise Exception(f"Error parsing LLM's JSON response: {str(result['parsing_error'])}")
                else:
                    # if we got neither an error nor a parsed response, raise an exception
                    raise Exception("Error parsing LLM's JSON response: no parsed response or error message")
            except Exception as e:
                self.logger.error(f"Error occurred asking question (attempt {attempt}): {str(e)}")
                if attempt < self.max_retries:
                    # wait before retrying
                    await asyncio.sleep(5)
                else:
                    # maximum attempts reached, return error
                    result_dict["result"] = "error"
                    result_dict["error"] = f"Max retries reached on question. Last error: {str(e)}"
                    return result_dict

        # copy parsed response (because we might update it later), record prompt and response in history
        response_dict = copy.deepcopy(result['parsed'])
        json_result = result['raw'].content
        if chat_history is not None:
            chat_history.append((question, json_result))
        full_history.append([question, json_result])

        # unless something goes wrong later, presume we're successful
        result_dict["result"] = "success"

        # ask follow-up questions
        for followup in followups:
            followup_result = await self.a_followup_question(**followup, response_dict=response_dict,
                                                             llm_chain=llm_chain, chat_history=chat_history)

            # if we actually asked a follow-up and got some kind of response, include it in the history
            if (followup_result["result"] != "skipped" and followup_result["prompt"] is not None
                    and followup_result["response_json"] is not None):
                full_history.append([followup_result["prompt"], followup_result["response_json"]])

            if followup_result["result"] == "success" and followup_result["response"]:
                # if we actually got a parsed response with a non-empty dict, update the response dict
                response_dict.update(followup_result["response"])
                # continue to the next follow-up (if any)
            elif followup_result["result"] == "error":
                # otherwise, if we got an error, update the result dict to reflect that
                result_dict["error"] = followup_result["error"]
                result_dict["result"] = "error"
                # and break from the loop to return
                break

        # assemble and return results
        result_dict["response"] = response_dict
        result_dict["history"] = full_history
        return result_dict

    def run_evaluation_chain(self, task_system_prompt: str, question: str, followups: list[dict],
                             chat_history: list = None) -> dict:
        """
        Run an evaluation chain (synchronously).

        :param task_system_prompt: System prompt template to use for the evaluation chain. This can include the
            {survey_context} and {survey_locations} variables to include information about the survey context. It
            should specify a specific JSON format for all responses.
        :type task_system_prompt: str
        :param question: Initial question to ask, to begin the evaluation chain. This should include the
            {survey_excerpt} variable to include the appropriate excerpt being evaluated.
        :type question: str
        :param followups: List of follow-up questions to ask, based on the JSON response to the initial question.
            Should be a list of dicts, each with a value for each of the following keys:
            "condition_func": function that returns True when the follow-up should be asked and False when it
            shouldn't (should take the following parameters: response_dict: dict, condition_key: str,
            condition_value: value to check against the one in the response dict);
            "condition_key": the key in the response dict to check;
            "condition_value": the value to check against the one in the response dict;
            "prompt_template": the template for the follow-up question to ask (which can include variables from
            the response dict).
        :type followups: list[dict]
        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        # run asynchronous process synchronously
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(
            self.a_run_evaluation_chain(
                task_system_prompt=task_system_prompt,
                question=question,
                followups=followups,
                chat_history=chat_history
            )
        )

    async def a_followup_question(self, condition_func: callable, condition_key: str, condition_value,
                                  prompt_template: str, response_dict: dict, llm_chain: Runnable,
                                  chat_history: list = None) -> dict:
        """
        Ask a follow-up question (asynchronously).

        :param condition_func: Function to call to evaluate whether the follow-up should be asked (True) or not (False).
        :type condition_func: callable
        :param condition_key: Key to check in the response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in the response dictionary, according to the logic of condition_func.
        :type condition_value: Any
        :param prompt_template: Template for the follow-up question to ask (which can include variables from the
            response dict).
        :type prompt_template: str
        :param response_dict: Full response dictionary.
        :type response_dict: dict
        :param llm_chain: Runnable conversation chain to use for asking the follow-up question.
        :type llm_chain: Runnable
        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :return: A dict with result ("success", "error", or "skipped"), error (if result is "error"), prompt (a str),
            response_json (a str), and response (a dict).
        :rtype: dict
        """

        # initialize results
        result_dict = {"result": None, "error": None, "prompt": None, "response_json": None, "response": None}

        # check to see if we meet the criteria for this follow-up
        if condition_func(response_dict, condition_key, condition_value):
            # if so, format the question
            followup_question = prompt_template.format(**response_dict)
            prompt = followup_question.strip()
            result_dict["prompt"] = prompt

            # ask question to the LLM, retrying the appropriate number of times
            attempt = 0
            while True:
                attempt += 1
                try:
                    # ask our question and break on success
                    result = await llm_chain.ainvoke({"question": followup_question},
                                                     config={"configurable": {"session_id": "NA"}})
                    if 'raw' in result and result['raw'] is not None and result['raw'].content:
                        # if there's a raw response, save it for return
                        result_dict["response_json"] = result['raw'].content

                    if 'parsed' in result and result['parsed'] is not None:
                        # if we got a parsed result, break from retry loop
                        break
                    elif 'parsing_error' in result and result['parsing_error'] is not None:
                        # if there was a parsing error, raise an exception
                        raise Exception(f"Error parsing LLM's JSON response to follow-up: "
                                        f"{str(result['parsing_error'])}")
                    else:
                        # if we got neither an error nor a parsed response, raise an exception
                        raise Exception("Error parsing LLM's JSON response to follow-up: no parsed response "
                                        "or error message")
                except Exception as e:
                    self.logger.error(f"Error occurred asking follow-up (attempt {attempt}): {str(e)}")
                    if attempt < self.max_retries:
                        # wait before retrying
                        await asyncio.sleep(5)
                    else:
                        # maximum attempts reached, return error
                        result_dict["result"] = "error"
                        result_dict["error"] = f"Max retries reached on follow-up question. Last error: {str(e)}"
                        return result_dict

            # note parsed response, record in history
            result_dict["response"] = result['parsed']
            result_dict["result"] = "success"
            if chat_history is not None and result_dict["response_json"] is not None:
                chat_history.append((followup_question, result_dict["response_json"]))
        else:
            result_dict["result"] = "skipped"

        # return assembled result
        return result_dict


class EvaluationLens:
    """Class for instrument evaluation lens, which is used to conduct a particular type of evaluation."""

    evaluation_engine: EvaluationEngine
    lens_description: str
    task_system_prompt_template: str
    question_template: str
    followups: list[dict]
    evaluation_result: dict

    def __init__(self, task_system_prompt_template: str, question_template: str, followups: list[dict],
                 evaluation_engine: EvaluationEngine,
                 lens_description: str = "Evaluation lens with unknown description"):
        """
        Initialize evaluation lens.

        :param task_system_prompt_template: System prompt template to use for the evaluation chain. This can include the
            {survey_context} and {survey_locations} variables to include information about the survey context. It
            should specify a specific JSON format for all responses.
        :type task_system_prompt_template: str
        :param question_template: Initial question to ask, to begin the evaluation chain. This should include the
            {survey_excerpt} variable to include the appropriate excerpt being evaluated.
        :type question_template: str
        :param followups: List of follow-up questions to ask, based on the JSON response to the initial question.
            Should be a list of dicts, each with a value for each of the following keys:
            "condition_func": function that returns True when the follow-up should be asked and False when it
            shouldn't (should take the following parameters: response_dict: dict, condition_key: str,
            condition_value: value to check against the one in the response dict);
            "condition_key": the key in the response dict to check;
            "condition_value": the value to check against the one in the response dict;
            "prompt_template": the template for the follow-up question to ask (which can include variables from
            the response dict).
        :type followups: list[dict]
        :param evaluation_engine: Evaluation engine instance to use for conducting evaluation.
        :type evaluation_engine: EvaluationEngine
        :param lens_description: High-level description of the evaluation lens, sufficient for LLM-as-judge evaluation
            of results.
        :type lens_description: str
        """

        # initialize object from constructor parameters
        self.lens_description = lens_description
        self.task_system_prompt_template = task_system_prompt_template
        if evaluation_engine.extra_evaluation_instructions:
            self.task_system_prompt_template += (f"\n\nAlso follow these additional instructions:\n\n"
                                                 f"{evaluation_engine.extra_evaluation_instructions}")
        self.question_template = question_template
        self.followups = followups
        self.evaluation_engine = evaluation_engine
        self.evaluation_result = {}

    def evaluate(self, chat_history: list = None, **kwargs) -> dict:
        """
        Run an evaluation chain (synchronously).

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        # call evaluation engine to run evaluation chain
        self.evaluation_result = self.evaluation_engine.run_evaluation_chain(
            task_system_prompt=self.task_system_prompt_template.format(**kwargs),
            question=self.question_template.format(**kwargs),
            followups=self.followups, chat_history=chat_history)
        return self.evaluation_result
    
    async def a_evaluate(self, chat_history: list = None, **kwargs) -> dict:
        """
        Run an evaluation chain (asynchronously).

        :param chat_history: Chat history to use for the evaluation chain (or None for none).
        :type chat_history: list
        :param kwargs: Keyword arguments to use for formatting the task system prompt and question.
        :type kwargs: Any
        :return: A dict with result ("success" or "error"), error (if result is "error"), response (a dict),
            and history (a list with the full history of the evaluation chain, each item of which is a list with two
            strings, a prompt and a response).
        :rtype: dict
        """

        # call evaluation engine to run evaluation chain
        self.evaluation_result = await self.evaluation_engine.a_run_evaluation_chain(
            task_system_prompt=self.task_system_prompt_template.format(**kwargs),
            question=self.question_template.format(**kwargs),
            followups=self.followups, chat_history=chat_history)
        return self.evaluation_result

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
            result_to_format = self.evaluation_result[0]

        # raise error if asked to filter for importance
        if minimum_importance > 0:
            raise NotImplementedError("No base class implementation for filtering by importance.")

        # return empty string if no result is available to format
        if result_to_format is None:
            return ""

        # format result as a list of key-value pairs
        formatted_results = []
        for key, value in result_to_format.items():
            if isinstance(value, list):
                value = [str(item) for item in value]
                if len(value) == 0:
                    value_str = ""
                elif len(value) == 1:
                    value_str = str(value[0])
                else:
                    value_str = ', '.join(value)
                formatted_results.append(f"{key}: {value_str}")
            else:
                formatted_results.append(f"{key}: {value}")

        # return the formatted result with each key-value pair on its own line
        return '\n'.join(formatted_results)

    def standardize_result(self, result: dict | None = None) -> list[dict]:
        """
        Reorganize the evaluation result into a list of recommendations in a standardized format.

        :param result: Evaluation result to format (or None to use the evaluation_result attribute).
        :type result: dict | None
        :return: List of recommendations, each of which is a dict with the following keys: importance (int 1-5),
            replacement_original (str), replacement_suggested (str), explanation (str).
        :rtype: list[dict]
        """

        raise NotImplementedError("No base class implementation for standardize_result().")

    @staticmethod
    def condition_is_value(response_dict: dict, condition_key: str, condition_value: str) -> bool:
        """
        Check if a condition is met in a given response dictionary: "is value" (string match).

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: str
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        return condition_key in response_dict and response_dict[condition_key] == condition_value

    @staticmethod
    def condition_is_not_value(response_dict: dict, condition_key: str, condition_value: str) -> bool:
        """
        Check if a condition is met in a given response dictionary: "is not value" (string doesn't match).

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: str
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        return condition_key in response_dict and response_dict[condition_key] != condition_value

    @staticmethod
    def condition_is_in_list(response_dict: dict, condition_key: str, condition_value: str) -> bool:
        """
        Check if a condition is met in a given response dictionary: "is in list" (string in list of strings).

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: str
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        return condition_key in response_dict and response_dict[condition_key] in condition_value

    @staticmethod
    def condition_is_not_in_list(response_dict: dict, condition_key: str, condition_value: str) -> bool:
        """
        Check if a condition is met in a given response dictionary: "is not in list" (string not in list of strings).

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: str
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        return condition_key in response_dict and response_dict[condition_key] not in condition_value

    @staticmethod
    def condition_list_has_greater_than_elements(response_dict: dict, condition_key: str, condition_value: int) -> bool:
        """
        Check if a condition is met in a given response dictionary: "list has greater than n elements".

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: int
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        if condition_key in response_dict and isinstance(response_dict[condition_key], list):
            return len(response_dict[condition_key]) > condition_value
        return False

    @staticmethod
    def condition_list_has_greater_or_equal_elements(response_dict: dict, condition_key: str, condition_value: int) \
            -> bool:
        """
        Check if a condition is met in a given response dictionary: "list has greater than or equal to n elements".

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: int
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        if condition_key in response_dict and isinstance(response_dict[condition_key], list):
            return len(response_dict[condition_key]) >= condition_value
        return False

    @staticmethod
    def condition_list_has_less_than_elements(response_dict: dict, condition_key: str, condition_value: int) \
            -> bool:
        """
        Check if a condition is met in a given response dictionary: "list has less than n elements".

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: int
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        if condition_key in response_dict and isinstance(response_dict[condition_key], list):
            return len(response_dict[condition_key]) < condition_value
        return False

    @staticmethod
    def condition_list_has_less_or_equal_elements(response_dict: dict, condition_key: str, condition_value: int) \
            -> bool:
        """
        Check if a condition is met in a given response dictionary: "list has less than or equal to n elements".

        :param response_dict: Response dictionary to check.
        :type response_dict: dict
        :param condition_key: Key to check in response dictionary.
        :type condition_key: str
        :param condition_value: Value to check for in response dictionary.
        :type condition_value: int
        :return: True if condition is met, False otherwise.
        :rtype: bool
        """

        if condition_key in response_dict and isinstance(response_dict[condition_key], list):
            return len(response_dict[condition_key]) <= condition_value
        return False
