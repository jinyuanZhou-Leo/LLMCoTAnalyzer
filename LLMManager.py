from openai import OpenAI, Stream
from loguru import logger
import sys
import numpy as np

from openai.types.chat.chat_completion import ChatCompletion
from openai.types.chat.chat_completion_chunk import ChatCompletionChunk

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True)


class LLMManager:
    def __init__(
        self,
        model_name: str,
        system_prompt: str,
        api_url: str,
        api_key: str,
        stream: bool = True,
        extra_body: dict = {},
        context: dict = {},
    ) -> None:
        """
        Initialize the LLMManager with the model name, system prompt, and API key.
        """
        self.model_name = model_name
        self.system_prompt = system_prompt
        self.api_key = api_key
        self.api_url = api_url
        self.stream = stream
        self.extra_body = extra_body
        self.context: list = context or []

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=self.api_url,
            )
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            raise e
        else:
            logger.success("OpenAI client initialized successfully")

        self.initialize_context()

    def initialize_context(self):
        self.context = []
        self.__add_context("system", self.system_prompt)

    def set_thinking(self, enable_thinking: bool):
        """
        Set the thinking mode for the model.
        """
        self.extra_body["enable_thinking"] = enable_thinking
        logger.info(f"Thinking mode set to: {enable_thinking}")

    def __add_context(self, role: str, content: str):
        """
        Add a message to the context.
        """
        self.context.append({"role": role, "content": content})

    def __create_chat_completion(self, user_prompt: str) -> ChatCompletion | Stream[ChatCompletionChunk]:
        """
        Create a chat completion based on the user prompt given and the history context previously stored
        """
        self.__add_context("user", user_prompt)
        return self.client.chat.completions.create(
            model=self.model_name,
            messages=self.context,
            extra_body=self.extra_body,
            stream=self.stream,
        )

    def get_chat_completion(self, user_prompt: str, verbose: bool = False) -> dict:
        """
        Get the chat completion from the model.
        """
        try:
            completion = self.__create_chat_completion(user_prompt)
        except Exception as e:
            logger.error(f"Failed to get chat completion: {e}")
            raise e

        reasoning_content = ""
        answer_content = ""
        if self.extra_body.get("enable_thinking", True):
            # the model is running in thinking mode
            is_answering = False  # a flag indicates whether the model has already begin answering
            if not verbose:
                print("Thinking: ", end="")
            for chunk in completion:
                delta = chunk.choices[0].delta
                # collect the reasoning content
                if hasattr(delta, "reasoning_content") and delta.reasoning_content is not None:
                    if not is_answering:
                        if not verbose:
                            print(delta.reasoning_content, end="", flush=True)
                    reasoning_content += delta.reasoning_content

                # collect the answer content
                if hasattr(delta, "content") and delta.content:
                    if not is_answering:
                        if not verbose:
                            print(f"\n{self.model_name.upper()} - Response: ", end="")
                        is_answering = True
                    if not verbose:
                        print(delta.content, end="", flush=True)
                    answer_content += delta.content
        else:
            # Thinking mode is off
            for chunk in completion:
                if not chunk.choices:
                    if not verbose:
                        print("\nUsage:")
                        print(chunk.usage)
                    continue

                delta = chunk.choices[0].delta
                if hasattr(delta, "content") and delta.content:
                    if not verbose:
                        print(delta.content, end="", flush=True)

        # logger.debug(f"Reasoning content: {reasoning_content}")
        # logger.debug(f"Answer content: {answer_content}")
        self.__add_context("assistant", reasoning_content + answer_content)
        return {"reasoning": reasoning_content, "answer": answer_content}


class TextEmbeddingManager:
    def __init__(self, model_name: str, api_key: str, api_url: str, dimensions: int = 1024):
        """
        Initialize the TextEmbeddingManager with the API key and URL.
        """
        self.model_name = model_name
        self.api_key = api_key
        self.api_url = api_url
        self.dimensions = dimensions
        self.client = OpenAI(
            api_key=api_key,
            base_url=api_url,
        )

    def get_embedding(self, text: str):
        """
        Get the embedding for the given text.
        """
        try:
            completion = self.client.embeddings.create(
                model=self.model_name,
                input=text,
                dimensions=self.dimensions,
                encoding_format="float",
            )
        except Exception as e:
            logger.error(f"Failed to get embedding: {e}")
            raise e

        return np.array(completion.data[0].embedding)


if __name__ == "__main__":
    logger.warning("This module is not meant to be run directly.")
