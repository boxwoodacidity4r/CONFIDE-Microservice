# models/embedding_manager.py

import os
import torch
from transformers import AutoTokenizer, AutoModel
import openai

class EmbeddingManager:
    def __init__(self, model_type="graphcodebert", openai_api_key=None, device=None):
        """
        model_type: "graphcodebert" | "codet5" | "openai"
        """
        self.model_type = model_type.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        if self.model_type in ["graphcodebert", "codet5"]:
            self._init_hf_model()
        elif self.model_type == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for openai embeddings")
            openai.api_key = openai_api_key
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def _init_hf_model(self):
        if self.model_type == "graphcodebert":
            model_name = "microsoft/graphcodebert-base"
        elif self.model_type == "codet5":
            model_name = "Salesforce/codet5-base"
        else:
            raise ValueError("Invalid HF model")

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def encode(self, code_snippets):
        """
        code_snippets: List[str] or str
        return: tensor embedding or list of floats
        """
        if isinstance(code_snippets, str):
            code_snippets = [code_snippets]

        if self.model_type in ["graphcodebert", "codet5"]:
            return self._encode_hf(code_snippets)
        elif self.model_type == "openai":
            return self._encode_openai(code_snippets)

    def _encode_hf(self, code_snippets):
        with torch.no_grad():
            inputs = self.tokenizer(code_snippets, padding=True, truncation=True, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            # 使用 [CLS] token 或 mean pooling 作为 embedding
            embeddings = outputs.last_hidden_state.mean(dim=1)
            return embeddings.cpu()

    def _encode_openai(self, code_snippets):
        embeddings = []
        for snippet in code_snippets:
            response = openai.Embedding.create(
                input=snippet,
                model="text-embedding-3-large"
            )
            embeddings.append(response['data'][0]['embedding'])
        return embeddings


if __name__ == "__main__":
    # 测试
    code_example = "public void placeOrder() { System.out.println('Order placed'); }"
    
    manager = EmbeddingManager(model_type="graphcodebert")
    emb = manager.encode(code_example)
    print("GraphCodeBERT embedding shape:", emb.shape)

    # OpenAI 测试
    # manager = EmbeddingManager(model_type="openai", openai_api_key="YOUR_KEY")
    # emb = manager.encode(code_example)
    # print("OpenAI embedding length:", len(emb[0]))
