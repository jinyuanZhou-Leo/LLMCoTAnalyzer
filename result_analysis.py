import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import json
from loguru import logger
from tqdm import tqdm
from typing import Callable

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
    .groupby(["Model", "Question"])["Count"]
    .progress_apply(list)
    .unstack(level=1)
    .progress_apply(lambda x: x.dropna(), axis=1)
)

# post processing
if config["postProcessingMethod"]["intraQuestion"].lower() == "sum":
    result = result.map(sum)
elif config["postProcessingMethod"]["intraQuestion"].lower() == "mean":
    result = result.map(np.mean)
else:
    logger.critical(f"Invalid intraQuestion post processing method: {config['postProcessingMethod']['intraQuestion']}")
    raise ValueError(f"Invalid intraQuestion post processing method: {config['postProcessingMethod']['intraQuestion']}")

if config["postProcessingMethod"]["interQuestion"].lower() == "sum":
    result = result.apply(sum, axis=1)
elif config["postProcessingMethod"]["interQuestion"].lower() == "mean":
    result = result.apply(np.mean, axis=1)
else:
    logger.critical(f"Invalid interQuestion post processing method: {config['postProcessingMethod']['interQuestion']}")
    raise ValueError(f"Invalid interQuestion post processing method: {config['postProcessingMethod']['interQuestion']}")

result = result.reset_index()
result.columns = ["Model", "Value"]

logger.info(f"Finished post processing, the result is shown below: \n {result}")

# 绘制散点图
plt.figure(figsize=(10, 6))

# 提取模型大小的数值（例如从 "qwen3-0.6b" 提取 0.6）
model_sizes = result["Model"].str.extract(r"-(\d+\.?\d*)b").astype(float)[0]  # 正则匹配数字部分
logger.warning(model_sizes)

# 在正则提取后添加校验
if model_sizes.isna().any():
    invalid_models = result["Model"][model_sizes.isna()].tolist()
    logger.error(f"模型名称格式错误，无法提取参数规模: {invalid_models}")
    raise ValueError("模型名称需符合 '前缀-参数规模b' 格式，例如 'qwen3-32b'")

# 绘制散点（使用数值型 model_sizes 作为 x 轴数据，保持原标签显示模型名称）
plt.scatter(
    x=model_sizes,  # 数值型 x 轴数据（用于回归计算）
    y=result["Value"],
    s=100,
    c="skyblue",
    edgecolor="black",
    zorder=2,  # 确保散点在回归线之上
)

# 计算线性回归
slope, intercept = np.polyfit(model_sizes, result["Value"], 1)  # 1 表示一次线性回归
regression_line = np.poly1d([slope, intercept])  # 生成回归方程

# 绘制回归线（按模型大小排序避免线条混乱）
sorted_indices = model_sizes.argsort()  # 按模型大小排序的索引
plt.plot(
    model_sizes[sorted_indices],  # 排序后的 x 轴数值
    regression_line(model_sizes[sorted_indices]),  # 对应的回归值
    color="red",
    linestyle="--",
    linewidth=2,
    label=f"Regression Line (y = {slope:.2f}x + {intercept:.2f})",
    zorder=1,  # 回归线在底层
)

# 设置标签（x轴仍显示模型名称，需调整刻度）
plt.xticks(ticks=model_sizes, labels=result["Model"], rotation=30)  # 用数值位置对应模型名称标签
plt.xlabel("Model Size", fontsize=12)
plt.ylabel("Self-verification Behavior Indicator", fontsize=12)
plt.title("Model Size vs Self-verification Behavior Indicator", fontsize=14)
plt.grid(linestyle="--", alpha=0.5)
plt.legend()  # 显示回归线图例
plt.tight_layout()
plt.show()
