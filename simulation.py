from semantic_analysis import SemanticChunks
from LLMManager import LLMManager, TextEmbeddingManager
from loguru import logger
from textClassificationSupervised import TextClassifier
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from datetime import datetime
import os
import csv

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)
logger.add("simulation.log", level="TRACE", rotation="150KB")

class Simulation:

    def __init__(
        self,
        model_list: list[str],
        repetition: int,
        question_list: list[str],
        system_prompt: str,
        filter_concept: list = None,
        embedding_method: str = "SupervisedClassification",
        output_path: str = f"simulation_results-{datetime.now().strftime("%m-%d-%H:%M:%S")}.csv",
    ):
        self.model_list = model_list
        self.repetition = repetition
        self.question_list = question_list
        self.system_prompt = system_prompt
        self.concept = filter_concept
        self.embedding_method = embedding_method
        self.output_path = output_path
        if self.embedding_method == "TextEmbedding":
            self.model = self.__init_text_embedding_model()
        elif self.embedding_method == "SBERT":
            self.model = self.__init_SBERT_model()
        elif self.embedding_method == "SupervisedClassification":
            self.model = self.__init_supervised_model()
        else:
            raise ValueError(f"Invalid method: {embedding_method}.")

    def __init_text_embedding_model(self) -> TextEmbeddingManager | None:
        try:
            return TextEmbeddingManager(
                model_name="text-embedding-v3",
                api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                dimensions=1024,
            )
        except Exception as e:
            logger.error(f"Failed to initialize text_embedding_model: {e}")
            return None

    def __init_SBERT_model(self) -> SentenceTransformer:
        return SentenceTransformer("all-MiniLM-L6-v2")
        # return SentenceTransformer("Alibaba-NLP/gte-multilingual-base", trust_remote_code=True)

    def __init_supervised_model(self) -> TextClassifier:
        return TextClassifier(
            model_name="Alibaba-NLP/gte-multilingual-base",
            train_batch_path="train.csv",
            eval_batch_path="val.csv",
            batch_size=16,
        )

    def start_simulation(self):
        with open(self.output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["Model", "Question", "Repetition", "Count", "Reasoning", "Timestamp"]
            )
            writer.writeheader()
            with tqdm(
                desc="Simulation Progress",
                leave=False,
                position=0,
                total=len(self.model_list) * len(self.question_list) * self.repetition,
                bar_format="{l_bar}{bar} | {percentage:3.1f}% {r_bar}",
            ) as overallpbar:
                for model_info in self.model_list:
                    model_name = model_info[0]
                    api_url = model_info[1]
                    api_key = os.getenv(model_info[2])
                    model = LLMManager(
                        model_name=model_name, system_prompt=self.system_prompt, api_url=api_url, api_key=api_key
                    )
                    model.set_thinking(enable_thinking=True)
                    with tqdm(
                        total=len(self.question_list) * self.repetition,
                        desc=f"Processing {model_name}",
                        position=1,
                        leave=True,
                    ) as modelpbar:
                        for question_index, question in enumerate(self.question_list):
                            for i in range(self.repetition):
                                model.initialize_context()  # clear all the context
                                logger.warning(
                                    f"Simulation Start | Model:{model_name}, Question: {question_index+1}, Repetition: {i + 1}"
                                )
                                chat_completion = model.get_chat_completion(question, verbose=True)
                                reasoning = SemanticChunks(chat_completion["reasoning"], model=self.model)
                                outliers_cnt = reasoning.count_concept(self.concept)
                                logger.success(f"Finished | Outliers_Cnt: {outliers_cnt}")
                                writer.writerow(
                                    {
                                        "Model": model_name,
                                        "Question": question,
                                        "Repetition": i + 1,
                                        "Count": outliers_cnt,
                                        "Reasoning": chat_completion["reasoning"],
                                        "Timestamp": datetime.now().isoformat(),
                                    }
                                )
                                csvfile.flush()  # 确保实时写入
                                modelpbar.update(1)
                                overallpbar.update(1)
        logger.success(f"\n\n\nSimulation completed. Results saved to {self.output_path}.")

if __name__ == "__main__":
    model_list = [
        ["qwen3-0.6b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-1.7b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-4b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
        ["qwen3-8b", "https://dashscope.aliyuncs.com/compatible-mode/v1", "DASHSCOPE_API_KEY"],
    ]
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
    question_list = ["What is the integral of x^2?", "What is the derivative of (lnx)^2?"]
    system_prompt = "You are a helpful assistant."
    simulation = Simulation(
        model_list=model_list,
        repetition=2,
        question_list=question_list,
        system_prompt=system_prompt,
        embedding_method="SupervisedClassification",
    )
    # filter_concept=concepts
    # ! if we use supervised classification, we need to provide the concepts
    simulation.start_simulation()
