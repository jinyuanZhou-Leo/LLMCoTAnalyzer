import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from loguru import logger
from tqdm import tqdm
from typing import Callable


def process_each_question(df: pd.DataFrame, method: Callable):
    return df.map(method)


# read the simulation result
file_path = "dummydata.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    logger.critical(f"Failed to read the file {file_path}: {e}")

tqdm.pandas()

questions = df["Question"].unique()
question_id_map = {question: f"{i}" for i, question in enumerate(questions)}
result: pd.DataFrame = (
    df.replace({"Question": question_id_map})
    .groupby(["Model", "Question"])["Count"]
    .progress_apply(list)
    .unstack(level=1)  # 改为unstack问题层级
    .progress_apply(lambda x: x.dropna(), axis=1)
)


result = result.map(sum)
result = result.apply(np.mean, axis=1)
print(result)
