from sentence_transformers import SentenceTransformer

embed_model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2")
embed_model.save("sentence_transformer_model")
