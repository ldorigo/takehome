
import asyncio
import json
import logging
from typing import AsyncGenerator, Literal, TypedDict, overload

import openai

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.StreamHandler())


openai.api_key = "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"


class FunctionCallResponse(TypedDict):
    name: str
    arguments: str


class OpenaiChatMessage(TypedDict):
    role: Literal["user", "system", "assistant"]
    content: str


class ParameterDict(TypedDict):
    type: str | list[str]
    description: str


class ParametersDict(TypedDict):
    type: Literal["object"]
    properties: dict[str, ParameterDict]
    required: list[str]


class OpenAifunction(TypedDict):
    name: str
    description: str
    parameters: ParametersDict


async def get_response_openai(
    messages: list[OpenaiChatMessage],
) -> AsyncGenerator[str, None]:
    try:
        response = await openai.ChatCompletion.acreate(
            model="gpt-4-0613",
            n=1,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            messages=messages,
            stream=True,
        )
    except Exception as openai_exception:
        logger.error(
            f"Error in creating campaigns from openAI: {str(openai_exception)}"
        )
        raise openai_exception
        
    try:
        async for chunk in response:
            # logger.debug(f"Chunk: {chunk}")
            choices = chunk["choices"]
            if len(choices) > 1:
                logger.warning(f"More than one choice returned??: {choices}")
            current_content = choices[0]["delta"].get("content", "")
            logger.debug(f"Current content: {current_content}")
            yield current_content
    except Exception as e:
        logger.error(f"Error in streaming response: {str(e)}")
        raise e


@overload
async def get_response_openai_nonstream(
    messages: list[OpenaiChatMessage], functions: list[OpenAifunction], function_name: str
) -> FunctionCallResponse:
    ...


@overload
async def get_response_openai_nonstream(
    messages: list[OpenaiChatMessage],
    functions: None = None,
    function_name: None = None,
) -> str:
    ...


# non-streaming version:
async def get_response_openai_nonstream(
    messages: list[OpenaiChatMessage],
    functions: list[OpenAifunction] | None = None,
    function_name: str|None = None,
) -> str | FunctionCallResponse:
    while True:
        try:
            args = {
                "model": "gpt-4-0613",
                # "model": "gpt-3.5-turbo-0613",
                "n": 1,
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
                "messages": messages,
            }
            if functions is not None:
                args["functions"] = functions
                # Remind the model to only retur valid json.....
                messages[0]["content"] = messages[0]["content"] + "\n\n Remember to ONLY use valid json when calling functions!!! This means escaping newlines and double quotes!!!"
            if function_name is not None:
                args["function_call"] = {
                    "name": function_name,
                }
            # logger.info(f"Sending request to OpenAI: {args}")
            response = await openai.ChatCompletion.acreate(**args)
            logger.info("Got response from OpenAI")
            choices = response["choices"]
            if len(choices) > 1:
                logger.warning(f"More than one choice returned??: {choices}")
            if functions is not None:
                function_call = choices[0]["message"].get("function_call", "")
                if not function_call:
                    logger.warning(f"No function call returned??: {function_call}")
                # logger.debug(f"Function call: {function_call}")
                arguments = function_call.get("arguments", "")
                if not arguments:
                    logger.warning(f"No arguments returned??: {arguments}")
                arguments_parsed = json.loads(arguments.replace("\n\n", "\\n \\n"))
                return arguments_parsed
            current_content = choices[0]["message"].get("content", "")
            # logger.debug(f"Current content: {current_content}")
            return current_content
        except json.JSONDecodeError as json_error:
            logger.warning(f"JSON error, retrying: {str(json_error)}")
            # wait a bit and try again
        except Exception as openai_exception:
            if "Overload" in str(openai_exception):
                logger.warning("Overload error, retrying")
                # wait a bit and try again
                await asyncio.sleep(1)
            else:
                logger.error(
                    f"Error in creating campaigns from openAI: {str(openai_exception)}"
                )
                raise openai_exception
            
    # try:
    # except Exception as e:
    #     logger.error(f"Error in streaming response: {str(e)}")
    #     raise e
