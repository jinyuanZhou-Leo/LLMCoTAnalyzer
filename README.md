# LLMCoTAnalyzer 大模型思维链模拟+分析

## Introduction 简介
LLMCoTAnalyzer is a simple tool for inspecting the specified semantic features within the chain of though of the large language model. The tool is built on top of the [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base), [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).

LLMCoTAnalyzer 是一个思维链分析工具工具，用于检查大型语言模型思维链中的指定语义特征。该工具基于 [Alibaba-NLP/gte-multilingual-base](https://huggingface.co/Alibaba-NLP/gte-multilingual-base) 和 [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) 构建。

## Deployment 部署
**0. What you need 项目要求**
- Python 3.10+
- Git/Github CLI
- uv 
  - [Installing uv 安装uv](https://docs.astral.sh/uv/getting-started/installation/)
- Dashscope API Key 阿里云百炼API密钥
  - [Get Your Dashscope API Key 获取百炼API Key](https://bailian.console.aliyun.com/?tab=doc#/doc/?type=model&url=https%3A%2F%2Fhelp.aliyun.com%2Fdocument_detail%2F2840915.html&renderType=iframe)
  - You might need to configure the API Key to the environment variable 需要将API Key配置到环境变量中
    - Windows
      ```bash
        setx DASHSCOPE_API_KEY "YOUR_DASHSCOPE_API_KEY"
        ```

    - MacOS
        ```bash
        echo "export DASHSCOPE_API_KEY='YOUR_DASHSCOPE_API_KEY'" >> ~/.zshrc
        
        source ~/.zshrc
        ```
  - You can use other OpenAI compatible API, the core function should be work perfectly. **The ```TextEmbedding``` method may not work properly if you decided to use other API.** 您可以使用其他OpenAI兼容的API，核心功能应该可以正常工作。**如果您决定使用其他API，```TextEmbedding```方法可能无法正常工作。**

**1. Clone the git repository 克隆储存库**

Using one the following commands to fetch the project to the local storage:

使用以下两个命令的其中之一将项目克隆到本地储存:
```bash
gh repo clone jinyuanZhou-Leo/LLMCoTAnalyzer
```
```bash
git clone https://github.com/jinyuanZhou-Leo/LLMCoTAnalyzer.git
```

**2. Enter the project dictionary 进入项目目录**
```bash
cd LLMCoTAnalyzer
```

**3. Dependency and Environment Configuration 依赖与环境配置**

This project use ```uv``` the package manager. You can configure the environment and install the dependencies by running

此项目使用```uv```作为python包管理器， 你可以通过运行以下命令配置环境和安装依赖
```bash
uv sync
```

**4. Run the project locally 在本地运行项目**

Congrats! The configuration is all done😛


Then, You can run the project by running


大功告成， 你可以通过运行以下脚本启动项目
- ```simulation.py```
- ```simulationGUI.py```

