import asyncio
import argparse
import os
from dotenv import dotenv_values
from llama_index.core import Settings, VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from qdrant_client import models
from tqdm.asyncio import tqdm

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, generation_with_knowledge_retrieval

async def main(force_rebuild_embedding: bool):
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    )
    embedding = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embedding

    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=5, use_fp16=True)

     # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=force_rebuild_embedding)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        data = read_data("data")
        pipeline = build_pipeline(llm, embedding, vector_store=vector_store)
        # 暂时停止实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=0),
        )
        await pipeline.arun(documents=data, show_progress=True, num_workers=1)
        # 恢复实时索引
        await client.update_collection(
            collection_name=config["COLLECTION_NAME"] or "aiops24",
            optimizer_config=models.OptimizersConfigDiff(indexing_threshold=20000),
        )
        print(len(data))

    retriever = QdrantRetriever(vector_store, embedding, similarity_top_k=10)

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        result, eval_result = await generation_with_knowledge_retrieval(
            query["query"], retriever, llm, embedding, reranker=reranker
        )
        results.append(result)
        print(eval_result)

    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG system with embedding control")
    parser.add_argument("--force-rebuild", action="store_true", help="Force rebuild embedding index")
    args = parser.parse_args()

    asyncio.run(main(args.force_rebuild))