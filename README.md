# LLMCoTAnalyzer 大模型思维链模拟+分析

## Introduction 简介
LLMCoTAnalyzer is a simple tool for inspecting the specified semantic features within the chain of though of the large language model. The tool is built on top of the [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base), [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

LLMCoTAnalyzer 是一个思维链分析工具工具，用于检查大型语言模型思维链中的指定语义特征。该工具基于 [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) 和 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 构建。

## Deployment 部署
1. Clone the git repository:
    ```bash
    gh repo clone jinyuanZhou-Leo/LLMCoTAnalyzer
    ```
    or 
    ```bash
    git clone https://github.com/jinyuanZhou-Leo/LLMCoTAnalyzer.git
    ```

2. This project use ```uv``` the package manager. You can configure the environment and install the dependencies by running:
    ```bash
    uv sync
    ```

3. Then, You can run the project by running:
    - ```simulation.py```
    - ```simulationGUI.py```

## Configuration 配置