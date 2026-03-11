import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel
import numpy as np
import faiss
import random
from dataclasses import dataclass
from typing import List


@dataclass
class Document:
    id: str
    text: str
    metadata: dict = None


@dataclass
class SearchResult:
    document: Document
    score: float
    rank: int

class DenseEncoder(nn.Module):
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        super().__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transformer = AutoModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)


    def mean_pooling(self, token_embeddings, attention_mask):

        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()

        return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


    def encode(self, texts):

        if isinstance(texts,str):
            texts=[texts]

        embeddings=[]

        with torch.no_grad():

            for t in texts:

                inputs=self.tokenizer(
                    t,
                    return_tensors='pt',
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)

                outputs=self.transformer(**inputs)

                emb=self.mean_pooling(outputs.last_hidden_state,inputs['attention_mask'])

                emb=F.normalize(emb,p=2,dim=1)

                embeddings.append(emb.cpu().numpy())

        return np.vstack(embeddings)


class MMRDenseRetriever:

    def __init__(self,encoder):

        self.encoder=encoder
        self.documents=[]
        self.doc_embeddings=None
        self.index=None


    def build_index(self,documents):

        self.documents=documents

        texts=[d.text for d in documents]

        self.doc_embeddings=self.encoder.encode(texts)

        dim=self.doc_embeddings.shape[1]

        self.index=faiss.IndexFlatIP(dim)

        self.index.add(self.doc_embeddings.astype(np.float32))


    def search(self,query,k=10):

        q=self.encoder.encode(query)

        scores,idx=self.index.search(q.astype(np.float32),k)

        results=[]

        for rank,(i,s) in enumerate(zip(idx[0],scores[0])):

            results.append(SearchResult(

                document=self.documents[i],
                score=float(s),
                rank=rank+1

            ))

        return results


    def sample_negative_documents(self,n=2):

        return random.sample(self.documents,n)


    def build_training_contexts(

        self,
        query,
        final_k=5,
        noise_k=1,
        shuffle=False

    ):

        retrieved=self.search(query,k=final_k)

        clean_contexts=[r.document.text for r in retrieved]

        noisy_contexts=list(clean_contexts)

        negatives=self.sample_negative_documents(noise_k)

        for n in negatives:
            noisy_contexts.append(n.text)

        if shuffle:
            random.shuffle(noisy_contexts)

        return {

            "clean_contexts":clean_contexts,
            "noisy_contexts":noisy_contexts,
            "retrieved_items":retrieved

        }