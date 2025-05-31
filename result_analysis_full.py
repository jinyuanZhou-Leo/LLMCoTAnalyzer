import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from loguru import logger
from tqdm import tqdm
import statsmodels.api as sm
from datetime import datetime
from pathlib import Path
import re

CSV_PATH = Path("simulation_results-05-29-19:19:44.csv")


def get_timestamp(filename: str) -> str:
    """
    Extract timestamp from the filename if it exists
    Otherwise, return the timestamp for the current time
    Args:
        filename (str): the filename
    Returns:
        str: the timestamp in the format of "%m-%d-%H-%M-%S"
    """
    match = re.search(r"(\d{2}[./-]\d{2}[./-]\d{2}[.:/-]\d{2}[.:/-]\d{2})", filename)
    if match:
        timestamp: str = match.group(1)
        timestamp = timestamp.replace("/", "-").replace(".", "-").replace(":", "-")
        try:
            datetime.strptime(timestamp, "%m-%d-%H-%M-%S")
            return timestamp
        except Exception as e:
            logger.warning(f"Invalid timestamp detected in filename: {e}")
    else:
        timestamp = datetime.now().strftime(
            "%m-%d-%H-M-%S"
        )  # ! Avoid using ":" in the filename, window does not support it


with open("analysis_config.json", "r", encoding="utf-8") as f:
    config: dict = json.load(f)

# read the simulation result
data_path = (Path.cwd() / CSV_PATH).resolve()
try:
    df = pd.read_csv(str(data_path))
except Exception as e:
    logger.critical(f"Failed to read the file {data_path}: {e}")

tqdm.pandas()

questions = df["Question"].unique()
question_id_map = {question: f"{i}" for i, question in enumerate(questions)}
result: pd.Series = (
    df.replace({"Question": question_id_map})
    .groupby(["Size", "Question"])["Count"]
    .progress_apply(list)
    .unstack(level=1)
    .progress_apply(lambda x: x.dropna(), axis=1)
)
logger.debug(f"\n{result}")


# Melt and explode，从这里开始处理 result
df = result.reset_index().melt(id_vars="Size", var_name="Question", value_name="Counts")
df = df.explode("Counts").rename(columns={"Counts": "Count"})
df["Count"] = df["Count"].astype(int)

# 处理缺失值
df = df.dropna(subset=["Size", "Count"])

timestamp = get_timestamp(data_path.stem)
result_path = f"result_analysis/{timestamp}"
os.makedirs(result_path, exist_ok=True)

# Boxplot of Self-doubt counts by Size
plt.figure()
df.boxplot(column="Count", by="Size")
plt.xlabel("Model Size (Billion Params)")
plt.ylabel("Self-Doubt Count")
plt.title("Boxplot of Self-Doubt Counts by Model Size")
plt.suptitle("")  # remove default title
plt.savefig(f"{result_path}/boxplot.png")
plt.show()

# Scatter + regression line
plt.figure()
plt.scatter(df["Size"], df["Count"])
# Fit linear model
X = sm.add_constant(df["Size"])
model = sm.OLS(df["Count"], X).fit()
# Regression line
line_x = df["Size"]
line_y = model.params["const"] + model.params["Size"] * line_x
plt.plot(line_x, line_y)
plt.xlabel("Model Size (Billion Params)")
plt.ylabel("Self-Doubt Count")
plt.title("Scatter Plot with Linear Regression Fit")
plt.savefig(f"{result_path}/linearRegression.png")
plt.show()

# Print regression summary
print(model.summary())
with open(f"{result_path}/linearRegressionDetails.log", "w") as f:
    f.write(model.summary().as_text())
