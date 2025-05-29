import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
import os
from loguru import logger
from tqdm import tqdm
import statsmodels.api as sm
from datetime import datetime


with open("analysis_config.json", "r", encoding="utf-8") as f:
    config: dict = json.load(f)

# read the simulation result
file_path = "dummydata.csv"
try:
    df = pd.read_csv(file_path)
except Exception as e:
    logger.critical(f"Failed to read the file {file_path}: {e}")

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

timestamp = datetime.now().strftime("%m-%d-%H:%M:%S")  # ! No Windows Compatibility
path = f"result_analysis/{timestamp}"
os.makedirs(path, exist_ok=True)

# Boxplot of Self-doubt counts by Size
plt.figure()
df.boxplot(column="Count", by="Size")
plt.xlabel("Model Size (Billion Params)")
plt.ylabel("Self-Doubt Count")
plt.title("Boxplot of Self-Doubt Counts by Model Size")
plt.suptitle("")  # remove default title
plt.savefig(f"{path}/boxplot.png")
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
plt.savefig(f"{path}/linearRegression.png")
plt.show()

# Print regression summary
print(model.summary())
with open(f"{path}/linearRegressionDetails.log", "w") as f:
    f.write(model.summary().as_text())
