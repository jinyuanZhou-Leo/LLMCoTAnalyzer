from LLMManager import TextEmbeddingManager
from sentence_transformers import SentenceTransformer, util
from sklearn.neighbors import LocalOutlierFactor
from loguru import logger
from tqdm import tqdm
import os
import re
import numpy as np

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)


class SemanticChunks:
    def __init__(self, content: str):
        self.content = content
        self.chunks = self.__split_into_chunks(content)

    def __str__(self):
        return self.content

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

    def __detect_outliers_lof(self, data, n_neighbors=20, contamination=0.1):
        """
        使用 LOF 检测异常值。

        参数:
            data (list[list[float]] or np.ndarray): 输入数据，多维或一维。
            n_neighbors (int): 用于比较密度的邻居数量。
            contamination (float): 异常值的比例估计。

        返回:
            outliers: 异常值列表
            inliers: 非异常值列表
        """
        data = np.array(data).reshape(-1, 1)
        lof = LocalOutlierFactor(n_neighbors=n_neighbors, contamination=contamination)
        y_pred = lof.fit_predict(data)

        outliers = data[y_pred == -1]
        inliers = data[y_pred == 1]

        return outliers.tolist(), inliers.tolist()

    def get_similarity(self, concepts, method: str):
        if method == "SBERT":
            similarities = self.__get_similarity_SBERT(concepts)
        elif method == "TextEmbedding":
            similarities = self.__get_similarity_TextEmbedding(concepts)
        else:
            raise ValueError("Invalid method. Choose 'SBERT' or 'TextEmbedding'.")

        cnt = len(self.__detect_outliers_lof(similarities)[0])
        logger.info(f"{cnt} outliers detected.")

    def __get_similarity_SBERT(self, concepts):
        model = SentenceTransformer("all-MiniLM-L6-v2")
        chunks = self.chunks
        concept_embeddings = model.encode(concepts, normalize_embeddings=True)
        similarities = []
        for chunk in tqdm(chunks, desc="Processing Chunks with SBERT", total=len(chunks)):
            chunk_embedding = model.encode(chunk, normalize_embeddings=True)
            chunk_similarities = util.cos_sim(chunk_embedding, concept_embeddings)[0].tolist()
            # print the average similarity
            chunk_average_similarities = sum(chunk_similarities) / len(chunk_similarities)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Avg Sim: {chunk_average_similarities}")
            similarities.append(chunk_average_similarities)
        return similarities

    def __get_similarity_TextEmbedding(self, concepts):
        text_embedding_model = None
        try:
            text_embedding_model = TextEmbeddingManager(
                model_name="text-embedding-v3",
                api_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
                api_key=os.getenv("DASHSCOPE_API_KEY"),
                dimensions=512,
            )
        except Exception as e:
            logger.error(f"Failed to initialize text_embedding_model: {e}")

        logger.info(f"Calculating concept embeddings for {concepts}")
        concept_embeddings = np.array([text_embedding_model.get_embedding(c) for c in concepts])
        logger.info(f"concept_embeddings: {concept_embeddings}")
        chunks: list[str] = self.chunks
        similarities = []
        for chunk in tqdm(chunks, desc="Processing Chunks with TextEmbedding", total=len(chunks)):
            chunk_embedding = text_embedding_model.get_embedding(chunk)
            chunk_similarities = util.cos_sim(chunk_embedding, concept_embeddings)[0].tolist()
            chunk_average_similarities = sum(chunk_similarities) / len(chunk_similarities)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Avg Sim: {chunk_average_similarities}")
            similarities.append(chunk_average_similarities)
        return similarities


if __name__ == "__main__":
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
    test_CoT = """Is it right? Okay, the user is asking for the integral of x squared. Let me recall how to integrate polynomials. 
I remember that the integral of x^n is (x^(n+1))/(n+1) + C, as long as n isn't -1. 
So here, n is 2. Applying that formula, it should be x cubed divided by 3 plus a constant. 
Wait, let me double-check. If I take the derivative of x^3/3, I get 3x²/3 which simplifies to x². Yep, that works. 
So the integral of x² dx is (x³)/3 + C. I should also mention that the constant C is important because the derivative of a constant is zero, so there are infinitely many antiderivatives. 
The user might be a student learning calculus, so explaining the power rule clearly would help. Maybe they need this for homework or an exam. 
I should present the answer step by step to make it clear. Also, maybe they want to know why the formula works, but since the question is straightforward, keeping it concise but correct is best. 
Let me make sure I didn't mix up any exponents. No, 2+1 is 3, so that's right. Alright, that should be it."""
    test_semantic_chunk = SemanticChunks(test_CoT)
    test_semantic_chunk.get_similarity(concepts=concepts, method="TextEmbedding")
