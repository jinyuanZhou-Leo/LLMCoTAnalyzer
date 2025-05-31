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

## Configuration 配置
You can configure the simulation and result analysis by editing ```simulation_config.json``` and ```analysis_config.json```.

可以通过编辑 ```simulation_config.json``` 和 ```analysis_config.json``` 来配置模拟和结果分析。

### 基础配置
1. `model_list` (模型列表)
   - 类型: 对象数组
   - 描述: 要测试的模型配置列表
   - 参数：
     - `name`: 模型名称 (e.g. "qwen3-14b")
     - `api_url`: 模型API地址
     - `api_key`: API访问密钥的名称（**环境变量的名称，并非密钥本身！**）
     - `size`: 模型参数量（单位：十亿）

2. `question_list` (问题列表)
   - 类型: 字符串数组
   - 描述: 用于测试的预设问题集
   - 示例: ["What is the integral of x^2?", ...]

### 测试参数
3. `ask_when_unsure` (不确定时询问)
   - 类型: 布尔值
   - 默认: false
   - 描述: 当模型返回不确定时是否继续追问（**仅适用于CLI**）

4. `repetition` (重复次数)
   - 类型: 整数 
   - 默认: 5
   - 描述: 每个问题的重复测试次数

5. `system_prompt` (系统提示)
   - 类型: 字符串
   - 描述: 模型系统提示词

### 高级配置
6. `method` (测试方法)
   - 类型: 字符串
   - 当前值: "SupervisedClassification"
   - 描述: 使用的测试方法

7. `advanced_config` (高级参数)
   - 类型: 对象
   - 参数：
     - `temperature`: 生成温度 (0-1)
     - `top_p`: 核心采样概率 (0-1)