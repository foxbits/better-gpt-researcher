"""Context compression utilities for GPT Researcher.

This module provides classes for compressing and retrieving relevant
context from documents using embeddings and similarity filtering.

The compression pipeline:
1. Splits documents into chunks
2. Filters chunks by embedding similarity to the query
3. Returns the most relevant chunks as context

Classes:
    VectorstoreCompressor: Retrieves context from a vector store.
    ContextCompressor: Compresses raw documents using embedding similarity.
    WrittenContentCompressor: Compresses previously written content sections.
"""

import asyncio
import os
from typing import Optional

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..memory.embeddings import OPENAI_EMBEDDING_MODEL
from ..prompts import PromptFamily
from ..utils.costs import estimate_embedding_cost, estimate_tokens
from ..vector_store import VectorStoreWrapper
from .retriever import SearchAPIRetriever, SectionRetriever


MAX_EMBEDDING_TOKENS = 8192  # BGE-M3 max tokens - set to 8192 for compatibility with strict embedding models like BAAI/bge-m3


class VectorstoreCompressor:
    """Retrieves and compresses context from a vector store.

    Uses similarity search on an existing vector store to find
    relevant documents for a given query.

    Attributes:
        vector_store: The vector store wrapper to search.
        max_results: Maximum number of results to return.
        filter: Optional filter for vector store queries.
    """

    def __init__(
        self,
        vector_store: VectorStoreWrapper,
        max_results: int = 7,
        filter: Optional[dict] = None,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the VectorstoreCompressor.

        Args:
            vector_store: The vector store to search.
            max_results: Maximum number of results to return.
            filter: Optional filter dictionary for queries.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.vector_store = vector_store
        self.max_results = max_results
        self.filter = filter
        self.kwargs = kwargs
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5) -> str:
        """Get relevant context from the vector store.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.

        Returns:
            Formatted string of relevant document content.
        """
        results = await self.vector_store.asimilarity_search(query=query, k=max_results, filter=self.filter)
        return self.prompt_family.pretty_print_docs(results)


class IndividualEmbeddingFilter:
    """Filters documents by embedding similarity using individual embeddings.

    This filter embeds documents one-by-one to avoid context_length_exceeded errors
    with embedding models that have strict token limits (e.g., BAAI/bge-m3 at 8192).
    """

    def __init__(self, embeddings, similarity_threshold: float):
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    async def acompress_documents(
        self, documents: list[Document], query: str
    ) -> list[Document]:
        """Filter documents by similarity to query using individual embeddings.

        Oversized documents are split into smaller chunks before embedding to ensure
        all content is processed.

        Args:
            documents: List of documents to filter.
            query: Query to compare document relevance against.

        Returns:
            List of documents that meet the similarity threshold.
        """
        query_embedding = await asyncio.to_thread(self.embeddings.embed_query, query)
        relevant_docs = []
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)

        for doc in documents:
            doc_tokens = estimate_tokens(doc.page_content)

            if doc_tokens > MAX_EMBEDDING_TOKENS:
                sub_chunks = splitter.split_documents([doc])
                for sub_chunk in sub_chunks:
                    sub_tokens = estimate_tokens(sub_chunk.page_content)
                    if sub_tokens > MAX_EMBEDDING_TOKENS:
                        continue
                    try:
                        sub_embedding = await asyncio.to_thread(
                            self.embeddings.embed_query, sub_chunk.page_content
                        )
                        similarity = self._cosine_similarity(query_embedding, sub_embedding)
                        sub_chunk.metadata["similarity_score"] = similarity
                        sub_chunk.metadata["parent_title"] = doc.metadata.get("title", "")
                        relevant_docs.append(sub_chunk)
                    except Exception:
                        continue
            else:
                try:
                    doc_embedding = await asyncio.to_thread(
                        self.embeddings.embed_query, doc.page_content
                    )
                    similarity = self._cosine_similarity(query_embedding, doc_embedding)

                    if similarity >= self.similarity_threshold:
                        doc.metadata["similarity_score"] = similarity
                        relevant_docs.append(doc)
                except Exception:
                    continue

        return relevant_docs

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        dot = sum(a * b for a, b in zip(vec1, vec2))
        mag1 = sum(a * a for a in vec1) ** 0.5
        mag2 = sum(b * b for b in vec2) ** 0.5
        return dot / (mag1 * mag2 + 1e-8)


class ContextCompressor:
    """Compresses raw documents to extract relevant context.

    Uses embedding similarity to filter document chunks and return
    only the most relevant content for a given query.

    Attributes:
        documents: List of documents to compress.
        embeddings: Embedding model for similarity calculation.
        max_results: Maximum number of results to return.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(
        self,
        documents,
        embeddings,
        max_results: int = 5,
        prompt_family: type[PromptFamily] | PromptFamily = PromptFamily,
        **kwargs,
    ):
        """Initialize the ContextCompressor.

        Args:
            documents: List of documents to compress.
            embeddings: Embedding model instance.
            max_results: Maximum number of results to return.
            prompt_family: Prompt family for formatting output.
            **kwargs: Additional keyword arguments.
        """
        self.max_results = max_results
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = float(os.environ.get("SIMILARITY_THRESHOLD", 0.35))
        self.prompt_family = prompt_family

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> str:
        """Get relevant context from documents asynchronously.

        Optimization: Skip expensive compression pipeline for small document sets.
        When documents are already concise, directly use them without embedding-based filtering.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            Formatted string of relevant document content.
        """
        total_chars = sum(len(str(doc.get('raw_content', ''))) for doc in self.documents)
        chunk_threshold = int(os.environ.get("COMPRESSION_THRESHOLD", "8000"))

        if total_chars < chunk_threshold and len(self.documents) <= max_results:
            direct_docs = [
                Document(
                    page_content=doc.get('raw_content', ''),
                    metadata=doc
                )
                for doc in self.documents[:max_results]
            ]
            return self.prompt_family.pretty_print_docs(direct_docs, max_results)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
        all_chunks = splitter.split_documents([
            Document(page_content=doc.get('raw_content', ''), metadata=doc)
            for doc in self.documents
        ])

        embedding_filter = IndividualEmbeddingFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold
        )

        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        relevant_docs = await embedding_filter.acompress_documents(all_chunks, query)

        relevant_docs.sort(key=lambda d: d.metadata.get("similarity_score", 0), reverse=True)
        return self.prompt_family.pretty_print_docs(relevant_docs[:max_results], max_results)


class WrittenContentCompressor:
    """Compresses previously written content sections.

    Specialized compressor for finding relevant sections from
    previously written report content, preserving section titles
    and structure.

    Attributes:
        documents: List of written content sections.
        embeddings: Embedding model for similarity calculation.
        similarity_threshold: Minimum similarity score for inclusion.
    """

    def __init__(self, documents, embeddings, similarity_threshold: float, **kwargs):
        """Initialize the WrittenContentCompressor.

        Args:
            documents: List of written content sections.
            embeddings: Embedding model instance.
            similarity_threshold: Minimum similarity score for inclusion.
            **kwargs: Additional keyword arguments.
        """
        self.documents = documents
        self.kwargs = kwargs
        self.embeddings = embeddings
        self.similarity_threshold = similarity_threshold

    async def async_get_context(self, query: str, max_results: int = 5, cost_callback=None) -> list[str]:
        """Get relevant written content sections asynchronously.

        Args:
            query: The search query.
            max_results: Maximum number of results to return.
            cost_callback: Optional callback for tracking embedding costs.

        Returns:
            List of formatted section strings.
        """
        sections = [
            Document(
                page_content=doc.get("written_content", ""),
                metadata={"section_title": doc.get("section_title", "")}
            )
            for doc in self.documents
        ]

        embedding_filter = IndividualEmbeddingFilter(
            embeddings=self.embeddings,
            similarity_threshold=self.similarity_threshold
        )

        if cost_callback:
            cost_callback(estimate_embedding_cost(model=OPENAI_EMBEDDING_MODEL, docs=self.documents))

        relevant_docs = await embedding_filter.acompress_documents(sections, query)

        relevant_docs.sort(key=lambda d: d.metadata.get("similarity_score", 0), reverse=True)
        return self._pretty_docs_list(relevant_docs, max_results)

    def _pretty_docs_list(self, docs: list[Document], top_n: int) -> list[str]:
        return [
            f"Title: {d.metadata.get('section_title')}\nContent: {d.page_content}\n"
            for i, d in enumerate(docs)
            if i < top_n
        ]