from LLMManager import TextEmbeddingManager
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import IsolationForest
from loguru import logger
from joblib import Memory
from tqdm import tqdm
import os
import re
import numpy as np

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)
memory = Memory(location="./cachedir", verbose=0)

class SemanticChunks:

    def __init__(self, content: str, method: str = "TextEmbedding"):
        self.content = content
        self.chunks = self.__split_into_chunks(content)
        self.method = method
        if self.method == "TextEmbedding":
            self.text_embedding_model = self.__init_text_embedding_model()
        elif self.method == "SBERT":
            self.sbert_model = self.__init_SBERT_model()
        else:
            raise ValueError(f"Invalid method: {method} Choose from 'TextEmbedding' or 'SBERT'")

    def __str__(self):
        return self.content

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

    def __split_into_chunks(self, text: str):
        """
        按句号、问号、感叹号分句；也可以按换行分段。
        """
        # 保留分隔符，方便后续理解
        chunks = re.split("([。！？.!?\n])", text)
        # 合并分隔符和前句
        sentences = []
        for i in range(0, len(chunks) - 1, 2):
            sent = chunks[i].strip() + chunks[i + 1]
            if sent.strip():
                sentences.append(sent.strip())
        return sentences

    def __detect_outliers_isolation_forest(self, data):
        data = np.array(data).reshape(-1, 1)
        model = IsolationForest(contamination="auto")
        preds = model.fit_predict(data)

        # 分离正常值和异常值
        normal_data = data[preds == 1].flatten()
        outlier_data = data[preds == -1].flatten()

        # 只保留右侧（比正常最大值大的异常）
        max_normal = np.max(normal_data)
        right_outliers = [x for x in outlier_data if x > max_normal]

        return right_outliers

    def identify_concept(self, concepts: list) -> list:
        similarities = self.get_similarity(concepts)
        outliers = self.__detect_outliers_isolation_forest(similarities)
        logger.info(f"{len(outliers)} outliers detected: {outliers}")
        return outliers

    def get_similarity(self, concepts):
        if self.method == "SBERT":
            return self.__get_similarity_SBERT(concepts)
        elif self.method == "TextEmbedding":
            return self.__get_similarity_TextEmbedding(concepts)
        else:
            raise ValueError("Invalid method. Choose 'SBERT' or 'TextEmbedding'.")

    def __get_similarity_SBERT(self, concepts):
        concept_embeddings = self.get_concept_embeddings(concepts, self.sbert_model)
        similarities = []
        for chunk in tqdm(self.chunks, desc="Processing Chunks with SBERT", leave=True, total=len(self.chunks)):
            chunk_embedding = self.sbert_model.encode(chunk, normalize_embeddings=True)
            chunk_similarities = util.cos_sim(chunk_embedding, concept_embeddings)[0].tolist()
            # print the average similarity
            chunk_compositive_similarities = self.amplified_sum(chunk_similarities)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Compositive Sim: {chunk_compositive_similarities}")
            similarities.append(chunk_compositive_similarities)
        return self.__normalize(similarities)

    def amplified_sum(self, arr):
        global concepts
        tmp = [i ** (1 / 4) for i in range(len(arr))]
        return sum(arr)

    def __normalize(self, arr: list) -> list:
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min()).tolist()

    @staticmethod
    @memory.cache
    def get_concept_embeddings(concepts: list[str], embedding_model):
        """Text Embedding method"""
        if isinstance(embedding_model, TextEmbeddingManager):
            return np.array([embedding_model.get_embedding(c) for c in concepts])
        elif isinstance(embedding_model, SentenceTransformer):
            return embedding_model.encode(concepts, normalize_embeddings=True)

    def __get_similarity_TextEmbedding(self, concepts):
        concept_embeddings = self.get_concept_embeddings(concepts, self.text_embedding_model)
        similarities = []
        chunk_embeddings = []
        for chunk in tqdm(self.chunks, desc="Text 2 Vec", leave=True, total=len(self.chunks)):
            chunk_embeddings.append(self.text_embedding_model.get_embedding(chunk))
        for chunk_embedding in tqdm(
            chunk_embeddings, desc="Processing Chunks Similarity", leave=True, total=len(chunk_embeddings)
        ):
            chunk_similarities = util.cos_sim(chunk_embedding, concept_embeddings)[0].tolist()
            # chunk_average_similarities = sum(chunk_similarities) / len(chunk_similarities)
            chunk_compositive_similarities = self.amplified_sum(chunk_similarities)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Compositive Sim: {chunk_compositive_similarities}")
            similarities.append(chunk_compositive_similarities)
        return self.__normalize(similarities)


if __name__ == "__main__":
    concepts = [
        "Is it right",
        "There is a mistake in my answer",
        "I got confused",
        "The previous answer is incorrect",
        "Am I right",
        "I was wrong",
        "What if I'm wrong",
        "I made a mistake",
        "Something doesn't add up",
        "I should double-check",
        "I should verify",
        "Check again",
    ]
    test_CoT = """ Okay, the user is asking for the integral of x squared. Let me recall how to integrate polynomials.
    I remember that the integral of x^n is (x^(n+1))/(n+1) + C, as long as n isn't -1.
    So here, n is 2. Applying that formula, it should be x cubed divided by 3 plus a constant.
    Wait, let me double-check. If I take the derivative of x^3/3, I get 3x²/3 which simplifies to x². Yep, that works.
    So the integral of x² dx is (x³)/3 + C. I should also mention that the constant C is important because the derivative of a constant is zero, so there are infinitely many antiderivatives.
    The user might be a student learning calculus, so explaining the power rule clearly would help. Maybe they need this for homework or an exam.
    I should present the answer step by step to make it clear. Also, maybe they want to know why the formula works, but since the question is straightforward, keeping it concise but correct is best.
    Let me make sure I didn't mix up any exponents. No, 2+1 is 3, so that's right. Alright, that should be it."""
    test_semantic_chunk = SemanticChunks(content=test_CoT, method="TextEmbedding")
    outliers = test_semantic_chunk.identify_concept(concepts=concepts)
    logger.success(f"There are {len(outliers)}, Outliers: {outliers}")
    print(type(outliers))
