import os
import sys
from LLMManager import LLMManager
from loguru import logger

logger.remove()
logger.add(sys.stdout, level="INFO", colorize=True)

model = None
try:
    model = LLMManager(
        model_name="qwen3-4b",
        system_prompt="You are a helpful assistant.",
        api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        api_key=os.getenv("DASHSCOPE_API_KEY"),
        stream=True,
    )
except Exception as e:
    logger.error(f"Error occurred while creating LLMManager: {e}")

model.set_thinking(True)

while True:
    user_input = input("User: ")
    if user_input.lower() == "exit":
        break

    try:
        model.get_chat_completion(user_input)
    except Exception as e:
        logger.error(f"Error occurred while getting chat completion: {e}")
        continue
