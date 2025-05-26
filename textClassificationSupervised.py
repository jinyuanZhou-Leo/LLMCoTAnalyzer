import pandas as pd
from tqdm import tqdm
from loguru import logger
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

logger.remove()
logger.add(lambda msg: tqdm.write(msg, end=""), level="DEBUG", colorize=True)


class TextClassifier:

    def __init__(self, model_name, train_batch_path: str, eval_batch_path: str, batch_size: int = 8):
        self.model_name = model_name
        self.train_batch_path = train_batch_path
        self.eval_batch_path = eval_batch_path
        self.batch_size = batch_size
        self.__init_model()
        self.__train()

    def __init_model(self) -> None:
        if torch.backends.mps.is_available():
            self.device = "mps"
            logger.success("MPS device is detected, training is accelerated using Apple Neural Engine")
        elif torch.cuda.is_available():
            self.device = "cuda"
            logger.success("CUDA device is detected, training is accelerated using NVIDIA GPU")
        else:
            self.device = "cpu"
            logger.warning("No GPU device is detected, try to use cpu instead")

        logger.info("Initializing pretrained model and tokenizer...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = AutoModel.from_pretrained(self.model_name, trust_remote_code=True)
            self.model = self.model.to(self.device)
            self.model.eval()
        except Exception as e:
            raise RuntimeError(f"Failed to load model or tokenizer: {e}")
        else:
            logger.info("Model and tokenizer initialized successfully.")

    def __train(self):
        train_pbar = tqdm(total=100, desc="Training", position=0, leave=True)
        train_df = pd.read_csv(self.train_batch_path)  # 训练集
        val_df = pd.read_csv(self.eval_batch_path)  # 验证集

        # 1. 检查数据集是否存在
        texts_train = train_df["text"].tolist()
        y_train = train_df["label"].astype(int).tolist()
        texts_val = val_df["text"].tolist()
        y_val = val_df["label"].astype(int).tolist()
        train_pbar.update(10)

        # 2. 生成向量
        X_train = self.__embed(texts_train, self.batch_size)
        X_val = self.__embed(texts_val, self.batch_size)
        train_pbar.update(20)

        # 3. 训练带 class_weight 的逻辑回归
        self.clf = LogisticRegression(class_weight="balanced", max_iter=1500)
        self.clf.fit(X_train, y_train)
        train_pbar.update(50)

        # 4. 在验证集上评估
        y_pred = self.clf.predict(X_val)
        logger.debug(f"\n{classification_report(y_val, y_pred, zero_division=0)}")
        train_pbar.update(20)
        train_pbar.close()
        logger.success("Finished Training")

    def __embed(self, text_list, batch_size):
        embeddings = []
        with (
            torch.no_grad(),
            (
                torch.cuda.amp.autocast("cuda")
                if self.device == "cuda"
                else torch.amp.autocast(device_type=self.device, enabled=False)
            ),
        ):
            for i in range(0, len(text_list), batch_size):
                batch = text_list[i : i + batch_size]
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)

                out = self.model(**encoded).last_hidden_state  # (B, L, D)

                # 使用attention mask加权平均代替简单平均
                attention_mask = encoded["attention_mask"]
                input_mask_expanded = attention_mask.unsqueeze(-1).expand(out.size()).float()
                pooled = torch.sum(out * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

                embeddings.append(pooled)
        return torch.cat(embeddings).cpu().numpy()

    def get_prediction(self, text: str):
        X_new = self.__embed([text], self.batch_size)
        prob = np.around(self.clf.predict_proba(X_new)[:, 1], decimals=3).tolist()[0]
        label = 1 if prob >= 0.7 else 0
        return label, prob

    def get_predictions(self, text_list: list):
        X_new = self.__embed(text_list, self.batch_size)
        probs = np.around(self.clf.predict_proba(X_new)[:, 1], 3).tolist()
        labels = [1 if p >= 0.7 else 0 for p in probs]
        return labels, probs


if __name__ == "__main__":
    logger.warning("This is a demo script, this module is not meant to be run directly.")
    classifier = TextClassifier(
        model_name="Alibaba-NLP/gte-multilingual-base",
        train_batch_path="train.csv",
        eval_batch_path="val.csv",
        batch_size=8,
    )
    text = "000000 gives approx 0."
    label, prob = classifier.get_prediction(text)
    print(f"Text: {text}, Label: {label}, Probability: {prob}")

    text_list = ["Okay, let me try to resolve the problem", "Maybe I get it wrong", "Wait a minutes"]
    labels, probs = classifier.get_predictions(text_list)
    for text, label, prob in zip(text_list, labels, probs):
        print(f"Text: {text}, Label: {label}, Probability: {prob}")
