from LLMManager import TextEmbeddingManager
from textClassificationSupervised import TextClassifier
from sentence_transformers import SentenceTransformer, util
from sklearn.ensemble import IsolationForest
import csv
from loguru import logger
from tqdm import tqdm
import re
import numpy as np
import logging

logging.getLogger("transformers.modeling_utils").setLevel(
    logging.ERROR
)  # ! Surpass the warning of SentenceTransformer partially initializing

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="INFO", colorize=True)

class SemanticChunks:

    def __init__(
        self,
        content: str,
        model: SentenceTransformer | TextEmbeddingManager | TextClassifier,
        ask_when_unsure: bool = False,
    ):
        self.content = content
        self.chunks = self.__split_into_chunks(content)
        self.model = model
        self.ask_when_unsure = ask_when_unsure
        if isinstance(self.model, TextEmbeddingManager):
            self.method = "TextEmbedding"
        elif isinstance(self.model, SentenceTransformer):
            self.method = "SBERT"
        elif isinstance(self.model, TextClassifier):
            self.method = "SupervisedClassification"

    def __str__(self):
        return self.content

    @staticmethod
    def get_concept_embeddings(concepts: list[str], embedding_model):
        """Get the text embeddings for concepts list"""
        if isinstance(embedding_model, TextEmbeddingManager):
            return np.array([embedding_model.get_embedding(c) for c in concepts])
        elif isinstance(embedding_model, SentenceTransformer):
            return embedding_model.encode(concepts, normalize_embeddings=True)

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
                if not re.fullmatch(r"^[\d\W_]+$", sent.strip()):
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

    def count_concept(self, concepts: list) -> list:
        if self.method != "SupervisedClassification":
            similarities = self.get_similarity(concepts)
            outliers_cnt = self.__detect_outliers_isolation_forest(similarities)
        else:
            outliers_cnt = self.get_similarity()
        logger.info(f"{outliers_cnt} outliers detected.")
        return outliers_cnt

    def get_similarity(self, concepts: list = None):
        """
        The composited method to get the similarity of each chunk with the concepts.
        """
        if self.method == "SBERT":
            return self.__get_similarity_SBERT(concepts)
        elif self.method == "TextEmbedding":
            return self.__get_similarity_TextEmbedding(concepts)
        elif self.method == "SupervisedClassification":
            return self.__get_similarity_SupervisedClassification()
        else:
            raise ValueError("Invalid method. Choose 'SBERT', 'TextEmbedding' or 'SupervisedClassification'.")

    def __get_similarity_SupervisedClassification(self):
        """
        Get the amplified sum of cosine similarity of each chunk using the Alibaba-NLP/gte-multilingual-base model

        :return: count of chunk that are filtered out by the model

        """
        cnt = 0
        for chunk in tqdm(
            self.chunks, desc="Classifying chunks with gte-text-embedding", leave=True, total=len(self.chunks)
        ):
            label, prob = self.model.get_prediction(chunk)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Label: {label}, Prob: {prob}")
            if label == 1:
                cnt += 1
            if self.ask_when_unsure and 0.4 <= prob <= 0.7:
                logger.warning(f"Chunk: {chunk}")
                logger.warning(f"Predicted Prob: {prob}, Please check it manually (1|0)")
                user_answer = input("Please enter 1 for True, 0 for False: ")
                with open(self.model.train_batch_path, "a", newline="", encoding="utf-8") as f:
                    writer = csv.writer(f)
                    writer.writerow([chunk, int(user_answer)])

        return cnt

    def __get_similarity_SBERT(self, concepts):
        """
        Get the amplified sum of cosine similarity of each chunk using the Sentence Transformer model(all-MiniLM-L6-v2)

        :param concepts: List of concepts to compare with
        :return: List of amplified sum of cosine similarity for each chunk

        """
        concept_embeddings = SemanticChunks.get_concept_embeddings(concepts, self.model)
        similarities = []
        for chunk in tqdm(self.chunks, desc="Processing Chunks with SBERT", leave=True, total=len(self.chunks)):
            chunk_embedding = self.model.encode(chunk, normalize_embeddings=True)
            chunk_similarities = util.cos_sim(chunk_embedding, concept_embeddings)[0].tolist()
            # print the average similarity
            chunk_compositive_similarities = self.amplified_sum(chunk_similarities)
            logger.debug(f"Chunk: {chunk}")
            logger.debug(f"Compositive Sim: {chunk_compositive_similarities}")
            similarities.append(chunk_compositive_similarities)
        return self.__normalize(similarities)

    def amplified_sum(self, arr) -> float:
        tmp = [i ** (1 / 4) for i in range(len(arr))]
        return sum(arr)

    def __normalize(self, arr: list) -> list:
        arr = np.array(arr)
        return (arr - arr.min()) / (arr.max() - arr.min()).tolist()

    def __get_similarity_TextEmbedding(self, concepts):
        """
        Get the amplified sum of cosine similarity of each chunk using the text-embedding-v3 model from Alibaba API

        :param concepts: List of concepts to compare with
        :return: List of amplified sum of cosine similarity for each chunk

        """
        concept_embeddings = SemanticChunks.get_concept_embeddings(concepts, self.model)
        similarities = []
        chunk_embeddings = []
        for chunk in tqdm(self.chunks, desc="Text 2 Vec", leave=True, total=len(self.chunks)):
            chunk_embeddings.append(self.model.get_embedding(chunk))
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
    logger.warning("This is a demo script, this module is not meant to be run directly.")
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
    text_classifier = TextClassifier(
        model_name="Alibaba-NLP/gte-multilingual-base",
        train_batch_path="train.csv",
        eval_batch_path="val.csv",
        batch_size=16,
    )
    test_semantic_chunk = SemanticChunks(content=test_CoT, model=text_classifier)
    outliers_cnt = test_semantic_chunk.count_concept(concepts=concepts)
    logger.success(f"There are {outliers_cnt} outliers")
