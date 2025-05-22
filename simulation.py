from semantic_analysis import SemanticChunks
from LLMManager import LLMManager
from loguru import logger
from tqdm import tqdm
from datetime import datetime
import os
import csv

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)


class Simulation:

    def __init__(
        self,
        model_list: list[str],
        repetition: int,
        question_list: list[str],
        system_prompt: str,
        filter_concept: list,
        embedding_method: str = "TextEmbedding",
        output_path: str = f"simulation_results-{datetime.now().strftime("%m-%d-%H:%M:%S")}.csv",
    ):
        self.model_list = model_list
        self.repetition = repetition
        self.question_list = question_list
        self.system_prompt = system_prompt
        self.concept = filter_concept
        self.embedding_method = embedding_method
        self.output_path = output_path

    def start_simulation(self):
        with open(self.output_path, "w", newline="") as csvfile:
            writer = csv.DictWriter(
                csvfile, fieldnames=["Model", "Question", "Repetition", "Count", "Reasoning", "Timestamp"]
            )
            writer.writeheader()
            with tqdm(self.model_list, desc="Overall Simulation", leave=False, position=0) as overallpbar:
                for model_info in overallpbar:
                    model_name = model_info[0]
                    api_url = model_info[1]
                    api_key = os.getenv(model_info[2])
                    model = LLMManager(
                        model_name=model_name, system_prompt=self.system_prompt, api_url=api_url, api_key=api_key
                    )
                    model.set_thinking(enable_thinking=True)
                    total_iterations = len(self.question_list) * self.repetition
                    with tqdm(
                        total=total_iterations, desc=f"Processing {model_name}", position=1, leave=True
                    ) as modelpbar:
                        for question in self.question_list:
                            for i in range(self.repetition):
                                model.initialize_context()  # clear all the context
                                logger.warning(
                                    f"Simulation Start | Model:{model_name}, Question: {question}, Repetition: {i + 1}"
                                )
                                chat_completion = model.get_chat_completion(question, verbose=True)
                                reasoning = SemanticChunks(chat_completion["reasoning"])
                                outliers = reasoning.identify_concept(self.concept)
                                logger.success(f"Finished | Outliers_Cnt: {len(outliers)}")
                                writer.writerow(
                                    {
                                        "Model": model_name,
                                        "Question": question,
                                        "Repetition": i + 1,
                                        "Count": len(outliers),
                                        "Reasoning": chat_completion["reasoning"],
                                        "Timestamp": datetime.now().isoformat(),
                                    }
                                )
                                csvfile.flush()  # 确保实时写入
                                modelpbar.update(1)


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
        filter_concept=concepts,
    )
    simulation.start_simulation()
