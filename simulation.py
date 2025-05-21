from semantic_analysis import SemanticChunks
from LLMManager import LLMManager
from loguru import logger
from tqdm import tqdm

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)

model_list = ["qwen3-0.6b", "qwen3-4b", "qwen3-8b"]


class Simulation:
    def __init__(self, model_list, repetition, question_list, embedding_method: str = "TextEmbedding"):
        self.model_list = model_list
        self.repetition = repetition
        self.question_list = question_list
        self.embedding_method = embedding_method

    def start_simulation(self):
        pass
