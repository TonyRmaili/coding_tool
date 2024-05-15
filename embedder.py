import faiss
import ollama
import numpy as np
import os


models = ["mxbai-embed-large", "llama3", "nomic-embed-text"]
db_names = ["mxbai_index", "llama3_index", "nomic_index"]

class Embedder:

    def __init__(self,model):
        self.codebase_path = "./codebase/"
        self.database_path = "./vector_dbs/"
        self.files = self.list_files()

        if model =="mxbai-embed-large":
            self.d = 1024
            self.index_name = "mxbai_index"

        elif model =="llama3":
            self.d = 4096
            self.index_name = "llama3_index"

        elif model =="nomic-embed-text":
            self.d = 768
            self.index_name = "nomic_index"

        else:
            raise ValueError("Invalid model name")
        
        self.model = model
        self.init_index()
        
    
    def save_index(self):
        faiss.write_index(self.index, self.database_path+self.index_name)
        print(f"Index saved to {self.index_name}")

    def list_files(self):
        return [f for f in os.listdir(self.codebase_path) if os.path.isfile(os.path.join(self.codebase_path, f))]

    def load_file(self, file):
        with open(self.codebase_path+file, "r") as f:
            code = f.read()
            return code
        
    def init_index(self):
        if os.path.exists(self.database_path+self.index_name):
            self.index = faiss.read_index(self.database_path+self.index_name)
        else:
            self.index = faiss.IndexFlatIP(self.d)
            self.save_index()
        
    def embed_file(self, code_index=0):
        code = self.load_file(self.files[code_index])
        response = ollama.embeddings(model=self.model, prompt=code)
        embedding = np.array(response["embedding"]).astype('float32')
        if embedding.shape[0] != self.d:
            raise ValueError(f"Embedding dimension mismatch: expected {self.d}, got {embedding.shape[0]}")
        embedding = embedding.reshape(1, -1)
        self.index.add(embedding)
        self.save_index()

    def embed_codebase(self):
        for i in range(len(self.files)):
            self.embed_file(code_index=i)
            print(f"File {i} embedded")
        print("Codebase embedded")

    
    def query(self, prompt,k=1):
        response = ollama.embeddings(model=self.model, prompt=prompt)
        query_embedding = np.array(response["embedding"]).astype('float32').reshape(1, -1)
        
        D, I = self.index.search(query_embedding, k)
        return D, I        


if __name__ == "__main__":

    # for model in models:
    #     emb = Embedder(model)
    #     emb.embed_codebase()
    
    # emb = Embedder("mxbai-embed-large")
    # print(emb.index.ntotal)


    for model in models:
        emb = Embedder(model)
        D,I = emb.query("Which file has math functions?")
        print(f"---From {model}---- ")
        print("D",D)
        print("I",I)
        print("most relevent file",emb.files[I[0][0]])

    