import logging
from typing import List, Dict, Tuple, Optional, AsyncGenerator

from langchain.llms.base import BaseLLM
from langchain.embeddings.base import Embeddings
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers.string import StrOutputParser
from langchain_qdrant import QdrantVectorStore
from langchain_community.retrievers import BM25Retriever
from qdrant_client.http.models import Filter, FieldCondition, MatchAny
from qdrant_client import QdrantClient

import os
from pathlib import Path
from dotenv import load_dotenv
from collections import defaultdict
import heapq
import asyncio
from enum import Enum

from pydantic import ValidationError,BaseModel
import openai
from langchain_core.exceptions import OutputParserException

from components.prompts import (
    get_discovery_prompt_template,
    get_detailed_prompt_template,
    get_multi_query_prompt_template,
    get_query_analysis_prompt_template
)


load_dotenv()

class Route(str, Enum):
    discovery = "discovery"
    detailed = "detailed"

class RouteDecision(BaseModel):
    route: Route

class QueryAnalysis(BaseModel):
    top_n: int
    rewritten_queries: List[str]

class SimpleRetriever:
    def __init__(
        self,
        collection_name: str,
        llm: BaseLLM,
        embeddings: Embeddings,
    ):
        self.collection_name = collection_name
        self.embeddings = embeddings
        self.qdrant_client = self.initialize_qdrant_client()

        self.discovery_top_n = 5
        self.use_multi_query = False
        self.multi_query_n = 3
        self.multi_query_ret_n = 3

        self.initialize(llm=llm)

    def initialize(self, llm,
                   discovery_top_n: Optional[int] = None, 
                   use_multi_query: Optional[bool] = None, 
                   multi_query_n: Optional[int] = None, 
                   multi_query_ret_n: Optional[int] = None):

        if llm is not None:
            self.llm = llm
        if discovery_top_n is not None:
            self.discovery_top_n = discovery_top_n
        if use_multi_query is not None:
            self.use_multi_query = use_multi_query
        if multi_query_n is not None:
            self.multi_query_n = multi_query_n
        if multi_query_ret_n is not None:
            self.multi_query_ret_n = multi_query_ret_n
        self.vectorstore = self.initialize_vectorstore()

    def initialize_qdrant_client(self) -> QdrantClient:
        qdrant_url = os.getenv("qdrant_link")
        qdrant_api_key = os.getenv("qdrant_api")
        if not qdrant_url or not qdrant_api_key:
            raise ValueError("QDRANT_LINK and QDRANT_API must be set.")
        return QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    def initialize_vectorstore(self) -> QdrantVectorStore:
        return QdrantVectorStore(
            client=self.qdrant_client,
            collection_name=self.collection_name,
            embedding=self.embeddings,
        )

    def create_retriever(self, top_n: int = 5, filter_data: list = None):
        search_kwargs = {"k": top_n}
        
        if filter_data is not None:
            qdrant_filter = Filter(
                should=[
                    Filter(
                        must=[
                            FieldCondition(key="metadata.category", match=MatchAny(any=filter_data)),
                        ]
                    ),
                ]
            )
            search_kwargs["filter"] = qdrant_filter
        
        return self.vectorstore.as_retriever(search_kwargs=search_kwargs)

    def create_multi_queries(self, query: str, n: int = 4) -> List[str]:
        multi_query_prompt = get_multi_query_prompt_template().format(n=n, query=query)
        chain = PromptTemplate.from_template(multi_query_prompt) | self.llm | StrOutputParser()
        output = chain.invoke({"query": query, "n": n}).split('</think>')[-1]
        queries = [q.strip("- ").strip() for q in output.strip().split("\n") if q.strip()]
        return queries



    def analyze_query(self, query: str, rewritten_n: int = 1, default_n: int = 5) -> Tuple[int, List[str]]:
       
        if not query or not isinstance(query, str):
            logging.error("Invalid or empty query provided.")
            raise ValueError("Query must be a non-empty string.")

        # Ensure rewritten_n is positive
        rewritten_n = max(1, rewritten_n)

        # Check if LLM supports structured output
        if not hasattr(self.llm, "with_structured_output"):
            logging.error("LLM does not support structured output.")
            raise RuntimeError("LLM must support with_structured_output for QueryAnalysis.")

        # Define the prompt for query analysis
        prompt_str = get_query_analysis_prompt_template().format(query=query, rewritten_n=rewritten_n, default_n=default_n)
        logging.debug(f"Query analysis prompt: {prompt_str}")
        prompt = PromptTemplate.from_template(prompt_str)

        try:
            # Use structured output with QueryAnalysis
            analyzer_llm = self.llm.with_structured_output(QueryAnalysis)
            result = analyzer_llm.invoke(prompt.format(query=query, rewritten_n=rewritten_n, default_n=default_n))

            # Validate top_n
            top_n = max(1, result.top_n)  # Ensure top_n is positive

            # Validate and adjust rewritten_queries
            rewritten_queries = result.rewritten_queries
            if not isinstance(rewritten_queries, list) or len(rewritten_queries) < rewritten_n:
                logging.warning(f"Insufficient rewritten queries: {rewritten_queries}. Padding with original query.")
                rewritten_queries = (rewritten_queries[:rewritten_n] + [query] * (rewritten_n - len(rewritten_queries)))[:rewritten_n]

            return top_n, rewritten_queries

        except ValidationError as e:
            logging.error(f"Structured output validation failed: {e}")
            return default_n, [query] * rewritten_n
        except Exception as e:
            logging.error(f"Unexpected error in analyze_query: {e}")
            return default_n, [query] * rewritten_n

    
    async def generate_response_streaming(
        self,
        rewritten_queries: List[str],
        retriever,
        multi_query_retriever,
        prompt_type: str = 'discovery',
        info=None,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> AsyncGenerator[str, None]:

        if multi_query_retriever is not None:
            schemes = await multi_query_retriever.invoke(rewritten_queries)
        else:
            schemes = await asyncio.to_thread(retriever.invoke, rewritten_queries[0])

        context = "\n\n".join(
            "Scheme Name: {name}\nDetails: {details}\nSource URL: {url}".format(
                name=s.metadata.get('name', 'Unknown Scheme'),
                details=s.page_content,
                url=s.metadata.get('source_url', "https://www.myscheme.gov.in/schemes/{}".format(s.metadata.get('scheme_id','404')))
            )
            for s in schemes
        )

        history_str = ""
        if chat_history:
            history_str = "\n".join([f"{role}: {message}" for role, message in chat_history])


        if prompt_type == 'discovery':
            prompt_str = get_discovery_prompt_template().format(
                question=rewritten_queries[0],
                context=context,
                info=info,
                chat_history=history_str
            )
        else:
            prompt_str = get_detailed_prompt_template().format(
                question=rewritten_queries[0],
                context=context,
                info=info,
                chat_history=history_str
            )
        logging.debug(f"LLM prompt: {prompt_str}")
        prompt = prompt_str

        try:
            async for chunk in self.llm.astream(prompt):
                chunk_content = chunk.content if hasattr(chunk, 'content') else chunk
                yield chunk_content
        except openai.AuthenticationError as e:
            logging.error(f"OpenAI Authentication Error: {e}")
            yield "An OpenAI authentication error occurred. Please check your OPENAI_API_KEY in the .env file."
        except OutputParserException as e:
            logging.error(f"Output Parser Exception: {e}")
            yield "An error occurred while parsing the LLM's output. This might be due to an invalid model response or configuration."
        except Exception as e:
            logging.error(f"An unexpected LLM error occurred: {e}")
            yield f"An unexpected error occurred with the LLM: {e}. Please try again or check your LLM configuration."

        
    async def generate_streaming(
        self, 
        query: str,
        filter_con= None,
        llm: Optional[BaseLLM] = None,
        discovery_top_n: Optional[int] = None,
        use_multi_query: Optional[bool] = None,
        multi_query_n: Optional[int] = None,
        multi_query_ret_n: Optional[int] = None,
        info=None,
        chat_history: Optional[List[Tuple[str, str]]] = None
    ) -> AsyncGenerator[str, None]:

        self.initialize(
            llm=llm, 
            discovery_top_n=discovery_top_n,
            use_multi_query=use_multi_query,
            multi_query_n=multi_query_n,
            multi_query_ret_n=multi_query_ret_n
        )

        if use_multi_query:
            query_n = multi_query_n
        else:
            query_n = 1

        top_n, rewritten_queries = self.analyze_query(query, rewritten_n=query_n, default_n=discovery_top_n)
        
        is_discovery = top_n > 1
        prompt_type = 'discovery' if is_discovery else 'detailed'

        retriever = self.create_retriever(top_n=top_n, filter_data=filter_con)

        if len(rewritten_queries) > 1:
            multi_query_retriever = MultiQueryRetriever(
                retriever=self.create_retriever(top_n=multi_query_ret_n or top_n, filter_data=filter_con),
                top_k=top_n
            )
        else:
            multi_query_retriever = None

        async for token in self.generate_response_streaming(
            rewritten_queries=rewritten_queries,
            retriever=retriever,
            multi_query_retriever=multi_query_retriever,
            prompt_type=prompt_type,
            info=info,
            chat_history=chat_history
        ):
            yield token


class MultiQueryRetriever:
    def __init__(self, retriever, top_k: int = 5, rrf_k: int = 60):
        self.retriever = retriever
        self.top_k = top_k
        self.rrf_k = rrf_k

    async def invoke(self, queries: List[str]):
        all_docs = []
        doc_map = {}
        scores = defaultdict(float)

        retriever_top_n = self.retriever.search_kwargs['k']
        for query in queries:
            docs = await asyncio.to_thread(self.retriever.invoke, query)
            all_docs.append(docs)

        for doc_list in all_docs:
            for rank, doc in enumerate(doc_list[:retriever_top_n]):
                key = doc.page_content
                scores[key] += 1 / (self.rrf_k + rank)
                doc_map[key] = doc

        ranked_keys = heapq.nlargest(self.top_k, scores.items(), key=lambda x: x[1])
        return [doc_map[key] for key, _ in ranked_keys]
