from typing import List, Tuple
from langchain_google_genai import ChatGoogleGenerativeAI
from app.services.persistence.mongodb import AsyncMongoDBSaver
from langgraph.graph import MessagesState
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
from app.services.neo4j_service import Neo4jService
from langgraph.graph import END, StateGraph, START
from langchain_core.documents import Document
from dotenv import load_dotenv
from app.services.rate_limit import llm_rate_limiter
import os
from enum import Enum
load_dotenv()
class ContextType(Enum):
    BOTH = "both"
    CYPHER = "cypher"
    VECTOR = "documents"
    EMPTY = "empty"
class State(MessagesState):
    summary: str
    documents : List[Document]
    cypher_answer: str
    context_type: ContextType = ContextType.BOTH
class ChatService:
    def __init__(self, 
                 memory,
                 Neo4jService: Neo4jService
                 ):
        self.memory = memory
        self.Neo4jService = Neo4jService
        # Initialize the graph workflow as an instance method
        self.graph = self.graph_workflow().compile(checkpointer=self.memory)
        self.llm = ChatGoogleGenerativeAI(
            model=os.getenv("CHAT_MODEL","gemini-2.0-flash"),
            max_retries=10,
            rate_limiter=llm_rate_limiter
        )
        
    @classmethod
    async def create(cls):
        # Asynchronously set up the instance and return it
        url = os.getenv("MONGO_URL") or None
        db_name = os.getenv("MONGO_DB_NAME") or None

        # Use async with for the context manager
        async with AsyncMongoDBSaver.from_conn_info(
            url=url, db_name=db_name
        ) as memory:
            # Create an instance of ChatService
            Neo4jService = Neo4jService()
            return cls(memory, Neo4jService)
    
    async def __asummarize_conversation(self,state: State):
        
        # First, we get any existing summary
        summary = state.get("summary", "")

        # Create our summarization prompt 
        if summary:
            
            # A summary already exists
            summary_message = (
                f"This is summary of the conversation to date: {summary}\n\n"
                "Extend the summary by taking into account the new messages above:"
            )
            
        else:
            summary_message = "Create a summary of the conversation above:"

        # Add prompt to our history
        model = self.llm
        messages = state["messages"][-6:] + [HumanMessage(content=summary_message)]
        response = await model.ainvoke(messages)
        
        # Delete all but the 4 most recent messages
        # delete_messages = [RemoveMessage(id=m.id) for m in state["messages"][:-4]]
        
        return {"summary": response.content}
    
    def __should_continue(self,state: State):
    
        """Return the next node to execute."""
        
        messages = state["messages"]
        
        # If there are more than six messages, then we summarize the conversation
        if len(messages) > 6:
            return "summarize_conversation"
        return END
    
    async def __retrieve(self,state: State):
        question = state.get("messages", "")[-1].content
        retriever = await self.Neo4jService.get_output(question,k=10)
        return {"cypher_answer": retriever.cypher_answer, "documents": retriever.relate_documents}
    
    async def __esg_model(self,state: State):
        model = self.llm

        context_type = state.get("context_type", ContextType.BOTH)
        cypher_answer = ""
        docs = []
        
        if context_type == ContextType.CYPHER:
            cypher_answer = state.get("cypher_answer", "")   
        elif context_type == ContextType.VECTOR:
            docs = state.get("documents", [])
        elif context_type == ContextType.BOTH:
            cypher_answer = state.get("cypher_answer", "")
            docs = state.get("documents", [])
            
        system_prompt, user_prompt = self.__prompt_rag(state["messages"][-1].content,
                                                        docs,
                                                        cypher_answer)
        prompt_template = ChatPromptTemplate([
            system_prompt,
            MessagesPlaceholder("msgs"),
            user_prompt,
        ])
        rag_chain = prompt_template | model 
        messages = state["messages"]
        summary = state.get("summary", "")
        if summary:
            # Add summary to system message
            system_message = f"Summary of conversation earlier: {summary}"
            # Append summary to any newer messages
            messages = [SystemMessage(content=system_message)] + state["messages"]
        messages = trim_messages(
            state["messages"],
            max_tokens=90000,
            strategy="last",
            token_counter=self.llm,
            allow_partial=False,
        )
        response = await rag_chain.ainvoke({"msgs": messages[:-1]})
        return {"messages": response}
    def __prompt_rag(self,question:str,documents: List[Document],structure:str) -> Tuple[SystemMessage, HumanMessage]:
        context = "\n".join([doc.page_content for doc in documents])
        system_prompt = f"""
                You are an assistant for question-answering tasks.
                You answer the question directly without 'Context: ' or 'Answer: '.
            """
        user_prompt = f"""
            Question: {question}

            Use the provided context and structure to answer in a concise and direct manner, without repeating the question.
            If the context doesn’t provide enough information, use general knowledge to assist. If neither is sufficient, state that you don’t know the answer.

            Structure: {structure}
            Context: {context}
                """
        
        return SystemMessage(content=system_prompt), HumanMessage(content=user_prompt)
    
    async def delete_by_thread_id(self,thread_id: str):
        await self.memory.delete_by_thread_id(thread_id)
        
    def graph_workflow(self):
        builder = StateGraph(State)
        builder.add_node("retrieve", self.__retrieve)
        builder.add_node("esg_model", self.__esg_model)
        builder.add_node("summarize_conversation", self.__asummarize_conversation)
        builder.add_edge(START, "retrieve")
        builder.add_edge("retrieve", "esg_model")
        builder.add_conditional_edges("esg_model",self.__should_continue,{
            "summarize_conversation": "summarize_conversation",
            END: END
        })
        builder.add_edge("summarize_conversation", END)
        return builder
        