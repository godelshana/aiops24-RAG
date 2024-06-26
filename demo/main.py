import asyncio

from dotenv import dotenv_values
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.core.node_parser import MarkdownNodeParser
from llama_index.core.ingestion import IngestionPipeline
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import MetadataMode
from qdrant_client import models
from tqdm.asyncio import tqdm
import jieba

from pipeline.ingestion import build_pipeline, build_vector_store, read_data
from pipeline.qa import read_jsonl, save_answers
from pipeline.rag import QdrantRetriever, HybridRetriever, generation_with_knowledge_retrieval
from custom.transformation import CustomFilePathExtractor, CustomTitleExtractor

async def main():
    config = dotenv_values(".env")

    # 初始化 LLM 嵌入模型 和 Reranker
    llm = Ollama(
        model="qwen", base_url=config["OLLAMA_URL"], temperature=0, request_timeout=120
    )
    embeding = HuggingFaceEmbedding(
        model_name="BAAI/bge-m3",
        cache_folder="./",
        embed_batch_size=128,
    )
    Settings.embed_model = embeding

    reranker = FlagEmbeddingReranker(model="BAAI/bge-reranker-v2-m3", top_n=6, use_fp16=True)

    # 关键词检索
    data = read_data("data")
    transformation = [
        MarkdownNodeParser(include_metadata=True, include_prev_next_rel=True),
        CustomTitleExtractor(metadata_mode=MetadataMode.EMBED),
        CustomFilePathExtractor(last_path_length=4, metadata_mode=MetadataMode.EMBED),
    ]
    bm25Pipeline = IngestionPipeline(transformations=transformation)
    nodes = await bm25Pipeline.arun(documents=data)
    bm25Retriever = BM25Retriever.from_defaults(nodes=nodes,tokenizer=jieba.lcut, similarity_top_k=10)

    # 初始化 数据ingestion pipeline 和 vector store
    client, vector_store = await build_vector_store(config, reindex=False)

    collection_info = await client.get_collection(
        config["COLLECTION_NAME"] or "aiops24"
    )

    if collection_info.points_count == 0:
        pipeline = build_pipeline(llm, embeding, vector_store=vector_store)
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

    qdranRetriever = QdrantRetriever(vector_store, embeding, similarity_top_k=10)

    retriever = HybridRetriever(qdranRetriever, bm25Retriever)

    queries = read_jsonl("question.jsonl")

    # 生成答案
    print("Start generating answers...")

    results = []
    for query in tqdm(queries, total=len(queries)):
        result = await generation_with_knowledge_retrieval(
            query["query"], bm25Retriever, llm, # reranker=reranker
        )
        results.append(result)

    # 处理结果
    save_answers(queries, results, "submit_result.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
