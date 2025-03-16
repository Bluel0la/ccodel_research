from ts.torch_handler.base_handler import BaseHandler
from sentence_transformers import SentenceTransformer
import torch

class SentenceTransformerHandler(BaseHandler):
    def __init__(self):
        super().__init__()
        self.model = None

    def initialize(self, ctx):
        """Load the model."""
        self.model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")

    def preprocess(self, data):
        """Preprocess input data."""
        sentences = [item['body'] for item in data]
        return sentences

    def inference(self, inputs):
        """Run inference."""
        embeddings = self.model.encode(inputs, convert_to_tensor=True)
        return embeddings.cpu().tolist()

    def postprocess(self, results):
        """Return the embeddings."""
        return results
