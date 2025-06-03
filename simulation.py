from semantic_analysis import SemanticChunks
from LLMManager import LLMManager, TextEmbeddingManager
from loguru import logger
from textClassificationSupervised import TextClassifier
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Literal
import os
import json
import csv

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)
logger.add("simulation.log", level="INFO", rotation="3MB")


class Simulation:

    def __init__(
        self,
        model_list: list[dict],
        repetition: int,
        question_list: list[str],
        system_prompt: str,
        filter_concept: list[str] = None,
        embedding_method: Literal["TextEmbedding", "SBERT", "SupervisedClassification"] = "SupervisedClassification",
        output_path: str = f"simulation_results-{datetime.now().strftime("%m-%d-%H-%M-%S")}.csv",
        ask_when_unsure: bool = False,
        max_threads: int = 10,
    ):
        self.model_list = model_list
        self.repetition = repetition
        self.question_list = question_list
        self.system_prompt = system_prompt
        self.concept = filter_concept
        self.embedding_method = embedding_method
        self.output_path = output_path
        self.ask_when_unsure = ask_when_unsure

        if self.embedding_method == "TextEmbedding":
            self.model = self.__init_text_embedding_model()
        elif self.embedding_method == "SBERT":
            self.model = self.__init_SBERT_model()
        elif self.embedding_method == "SupervisedClassification":
            self.model = self.__init_supervised_model()
        else:
            raise ValueError(f"Invalid method: {embedding_method}.")

        self.max_threads = max_threads
        self.write_lock = threading.Lock()

    def __init_text_embedding_model(self) -> TextEmbeddingManager | None:
        try:
            return TextEmbeddingManager(
                model_name="text-embedding-v3",
                api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                dimensions=1024,
            )
        except Exception as e:
            logger.critical(f"Failed to initialize text_embedding_model: {e}")
            return None

    def __init_SBERT_model(self) -> SentenceTransformer:
        try:
            return SentenceTransformer("all-MiniLM-L6-v2")
            # return SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)
        except Exception as e:
            logger.critical(f"Failed to initialize SBERT model: {e}")

    def __init_supervised_model(self) -> TextClassifier:
        try:
            return TextClassifier(
                model_name="Alibaba-NLP/gte-multilingual-base",
                train_batch_path="train.csv",
                eval_batch_path="val.csv",
                batch_size=32,
            )
        except Exception as e:
            logger.critical(f"Failed to initialize supervised model: {e}")

    def process_simulation_task(self, model_config: dict, question_id, repetition):
        logger.info(f"Task | Model:{model_config["name"]}, Question: {question_id+1}, Repetition: {repetition+1}")
        try:
            model = LLMManager(
                model_name=model_config["name"],
                system_prompt=self.system_prompt,
                api_url=model_config["api_url"],
                api_key=os.getenv(model_config["api_key"]),
            )
        except Exception as e:
            logger.error(f"Failed to initialize {model["name"]}: {e}")
            return None

        model.set_thinking(True)
        model.initialize_context()

        chat_completion = model.get_chat_completion(self.question_list[question_id], verbose=True)
        reasoning = SemanticChunks(chat_completion["reasoning"], self.model, self.ask_when_unsure)

        return {
            "Model": model_config["name"],
            "Size": model_config["size"],
            "Question": self.question_list[question_id],
            "Repetition": repetition,
            "Count": reasoning.count_concept(self.concept),
            "Reasoning": chat_completion["reasoning"],
            "Timestamp": datetime.now().isoformat(),
        }

    def _write_result(self, future, writer, csvfile, pbar):
        try:
            result = future.result()
            with self.write_lock:
                writer.writerow(result)
                if pbar.n % 10 == 0:
                    csvfile.flush()
            pbar.update(1)
        except Exception as e:
            logger.error(f"Simulation task {future} failed: {e}")

    def start_simulation(self):
        with open(self.output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["Model", "Size", "Question", "Repetition", "Count", "Reasoning", "Timestamp"]
            )
            writer.writeheader()

            task_configs = [
                (model_config, i, j)
                for model_config in self.model_list
                for i in range(len(self.question_list))
                for j in range(self.repetition)
            ]

            with (
                ThreadPoolExecutor(max_workers=self.max_threads) as executor,
                tqdm(
                    desc="Simulation Progress",
                    leave=False,
                    position=0,
                    total=len(self.model_list) * len(self.question_list) * self.repetition,
                    bar_format="{l_bar}{bar} | {percentage:3.1f}% {r_bar}",
                ) as simulation_pbar,
            ):
                BATCH_SIZE = 50
                futures = []
                for config in task_configs:
                    # when there are more than BATCH_SIZE tasks in the queue
                    # we start to process the result of each task
                    if len(futures) >= BATCH_SIZE:
                        for future in as_completed(futures):
                            self._write_result(future, writer, csvfile, simulation_pbar)
                        futures = []
                    futures.append(executor.submit(self.process_simulation_task, *config))

                # process the remaining tasks, there might not be BATCH_SIZE tasks in the queue
                for future in as_completed(futures):
                    self._write_result(future, writer, csvfile, simulation_pbar)
        logger.success(f"\n\n\nSimulation completed. Results saved to {self.output_path}.")

    def terminate_simulation(self):
        raise ForceTerminateException("Simulation terminated by user.")


class ForceTerminateException(Exception):
    pass


if __name__ == "__main__":
    concepts = [
        "Is it right?",
        "Wait, maybe better to",
        "Maybe there is a mistake in my answer",
        "Maybe I got confused",
        "The previous answer is not correct",
        "Am I right?",
        "I realize now that I was wrong",
        "What if I'm wrong?",
        "Assuming that I made a mistake",
        "Something doesn't add up",
    ]
    with open("simulation_config.json", "r", encoding="utf-8") as f:
        config: dict = json.load(f)

    simulation = Simulation(
        model_list=config["model_list"],
        repetition=int(config["repetition"]),
        question_list=[question for question in config["question_list"]],
        system_prompt=config["system_prompt"],
        embedding_method=config.get("method", "SupervisedClassification"),
        ask_when_unsure=config.get("ask_when_unsure", False),
        filter_concept=config.get("filter_concept", None),
        max_threads=config.get("max_threads", 10),
    )
    # ! if we use supervised classification, we don't need to provide the concepts
    simulation.start_simulation()
