from pdf2image import convert_from_path
from concurrent.futures import ThreadPoolExecutor
import os
import base64
from io import BytesIO
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import asyncio
from langchain_core.documents import Document
from typing import List,AsyncGenerator
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser
from langchain.prompts import ChatPromptTemplate
from langchain_core.exceptions import OutputParserException
from langchain_text_splitters import RecursiveCharacterTextSplitter
from azure.core.credentials import AzureKeyCredential
from azure.ai.documentintelligence import DocumentIntelligenceClient
from azure.ai.documentintelligence.models import AnalyzeOutputOption, AnalyzeResult, ContentFormat
from azure.core.exceptions import HttpResponseError
import json
from tenacity import retry, stop_after_attempt, wait_fixed
from functools import wraps
import itertools
import aiofiles
class QAResponse(BaseModel):
    question: str
    answer: str
class ListQAResponse(BaseModel):
    list_answer: List[QAResponse]
    
def retry_on_exception(func=None, *, attempts: int = 3, delay: int = 1):
    """
    Decorator to retry a coroutine up to a specified number of times with a fixed delay if an exception occurs.

    Parameters:
        func (Optional[Callable]): The function to wrap. Allows decorator to work with or without parameters.
        attempts (int): Number of retry attempts (default=3).
        delay (int): Delay in seconds between retries (default=1).
    """
    if func is None:
        def wrapper_with_params(func):
            @wraps(func)
            async def wrapper(*args, **kwargs):
                for attempt in range(attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        if attempt < attempts - 1:
                            await asyncio.sleep(delay)
                        else:
                            raise e
            return wrapper
        return wrapper_with_params
    else:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            for attempt in range(attempts):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    if attempt < attempts - 1:
                        await asyncio.sleep(delay)
                    else:
                        raise e
        return wrapper


class FinetuneService:    
    def load_pdf_images(self,pdf_path, dpi=300,thread_count=10):
        images = convert_from_path(pdf_path, dpi=dpi,thread_count=thread_count, poppler_path='/usr/local/bin')
        images_base64 = []
        for i, image in enumerate(images):
            buffer = BytesIO()
            image.save(buffer, format="JPEG")
            buffer.seek(0)
            images_base64.append(base64.b64encode(buffer.read()).decode("utf-8"))
        return images_base64
    
    async def generate_explain_images(self, pdf_path, model:str ="gpt-4o-mini",
                                      thread_count=10, batch_size=50) -> List[Document]:
        images = self.load_pdf_images(pdf_path,thread_count=thread_count)
        tasks = []
        results = []
        for i, image in enumerate(images):
            tasks.append(asyncio.create_task(self.__image_model(image,model=model)))
            if len(tasks) == batch_size:
                results += await asyncio.gather(*tasks)
                tasks = []
        if len(tasks) > 0:
            results += await asyncio.gather(*tasks)
        return results

    async def __image_model(self, image, model:str ="gpt-4o-mini") -> Document:
        model: ChatOpenAI = ChatOpenAI(
            temperature=0,
            model=model,
        )
        msg = await model.ainvoke(
            [
                HumanMessage(
                    content=[
                        {"type": "text", "text": "Describe this image thoroughly and in detail, covering every visible aspect without summarizing. Include specific details about each part of the image.please not lazy answers "},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image}",
                            },
                        },
                    ]
                )
            ]
        )
        return Document(page_content=msg.content)
    
    async def generate_from_folder(self,
                                   folder_path,
                                   model:str ="gpt-4o-mini",
                                   thread_count=10, each_pdf_batch_size=50) -> List[Document]:
        tasks = []
        results = []
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    results += await self.generate_explain_images(os.path.join(root, file),
                                                                  model=model,
                                                                  thread_count=thread_count,
                                                                  batch_size=each_pdf_batch_size)
        return results
    
    async def generate_from_single_pdf(self, pdf_path,
                                       model:str ="gpt-4o-mini",
                                       thread_count=10, each_pdf_batch_size=50) -> List[Document]:
        return await self.generate_explain_images(pdf_path,
                                                  model=model,
                                                  thread_count=thread_count,
                                                  batch_size=each_pdf_batch_size)
        
    async def generate_QA_synthetic(self, chuck_context: str, number_of_questions: int) -> List[QAResponse]:
        
        model = ChatOpenAI(model="gpt-4o-mini")
        system_prompt = """You are a legal expert capable of creating questions and answers based on legal content.
        Please review the provided legal context and generate questions that a reader might have. Then, clearly answer those questions based on the given information, ensuring each question is self-contained and well-defined."""
        prompt_user = """
                        Context: {chuck_context}
                        Generate {number_of_questions} questions 
                        and ensure they meet the following personality traits:
                        Personality 1: The questions and answers will be used to fine-tune a model, so the answers may include examples when relevant. Do not guess the answers; base them strictly on the provided context, which is part of a document.
                        Respond in a valid JSON format.
                        {format_instructions}
                    """
        parser =  PydanticOutputParser(pydantic_object=ListQAResponse)
        prompt_template = ChatPromptTemplate([
        ("system", system_prompt),
        ("user", prompt_user)
        ],partial_variables={"format_instructions": parser.get_format_instructions()})
        chain = prompt_template | model | parser 
        try:
            response = await chain.ainvoke({"chuck_context":chuck_context, "number_of_questions":number_of_questions})
            return response.list_answer
        except OutputParserException:
            return []
        except Exception as e:
            return []
        
    async def split_documents_into_chunks(self, documents: List[Document], chunk_size: int = 4096, chunk_overlap: int = 1024):
        splitter = RecursiveCharacterTextSplitter(
                            chunk_size=chunk_size, 
                            chunk_overlap=chunk_overlap
                )
        return await asyncio.to_thread(splitter.split_documents, documents)
    
    def convert_to_message_format(self, responses: List[QAResponse],save_path: str):
        system = {"role": "system", "content": "You are an expert in Environmental, Social, and Governance (ESG) topics."}
        messages = []
        for response in responses:
            messages_format = { "messages": [] }
            messages_format["messages"].append(system)
            messages_format["messages"].append({"role": "user", "content": response.question})
            messages_format["messages"].append({"role": "assistant", "content": response.answer})
            messages.append(messages_format)
            
        if save_path:
             # Save messages to a JSONL file
            with open(save_path, 'w', encoding='utf-8') as file:
                for message in messages:
                    file.write(json.dumps(message, ensure_ascii=False) + '\n')
                    
        return messages
    
    def read_single_pdf_to_markdown(self, file_path: str) -> Document:
        endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT")
        key = os.getenv("DOCUMENT_INTELLIGENCE_KEY")
        credential = AzureKeyCredential(key)
        document_intelligence_client = DocumentIntelligenceClient(endpoint, credential)
        with open(file_path, "rb") as f:
                poller = document_intelligence_client.begin_analyze_document(
                    "prebuilt-read",
                    analyze_request=f,
                    content_type="application/octet-stream",
                    output_content_format=ContentFormat.MARKDOWN
                )
        result: AnalyzeResult = poller.result()
        content = result.content
        return Document(page_content=content)
    
    async def aread_folder_pdf_to_markdown(self, folder_path: str) -> AsyncGenerator[Document,None]:
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    try:
                        document = await self.aread_pdf_to_markdown(os.path.join(root, file))
                        yield document
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
                        
    @retry_on_exception                 
    async def aread_pdf_to_markdown(self, file_path: str) -> Document:
        return await asyncio.to_thread(self.read_single_pdf_to_markdown, file_path)
    
    async def generate_qa_from_documents(self, documents: List[Document], number_of_questions_each_doc: int = 10,
                                         batch_size: int = 50) -> List[QAResponse]:
        qa = []
        tasks = []
        for doc in documents:
            tasks.append(asyncio.create_task(self.generate_QA_synthetic(doc,number_of_questions=number_of_questions_each_doc)))
        if len(tasks) == batch_size:
            nested_results = await asyncio.gather(*tasks)
            flat_results = [item for sublist in nested_results for item in sublist]
            qa += flat_results
            tasks = []
        if len(tasks) > 0:
            nested_results = await asyncio.gather(*tasks)
            flat_results = [item for sublist in nested_results for item in sublist]
            qa += flat_results
        return qa
    
    async def flow_generate_train_data(self, folder_path: str, save_folder: str,
                                       number_of_questions_each_chuck_doc: int = 10,
                                       batch_size: int = 50):
        
        if not os.path.exists(folder_path):
            print("Folder path does not exist!")
            return

        if not os.path.exists(save_folder):
            os.makedirs(save_folder)
    
        for root, _, files in os.walk(folder_path):
            for file in files:
                if file.endswith(".pdf"):
                    try:
                        document = await self.aread_pdf_to_markdown(os.path.join(root, file))
                        chunked_documents = await self.split_documents_into_chunks([document])
                        qa = await self.generate_qa_from_documents(chunked_documents,
                                                                   number_of_questions_each_doc=number_of_questions_each_chuck_doc,
                                                                   batch_size=batch_size)
                        base_name = os.path.splitext(file)[0]
                        self.convert_to_message_format(qa, os.path.join(save_folder, f"{base_name}.jsonl"))
                    except Exception as e:
                        print(f"Failed to process {file}: {e}")
    
    async def merge_jsonl_files(self, folder_path: str, save_path: str):
        """
        Merges all `.jsonl` files in a given folder into a single `.jsonl` file.

        Parameters:
            folder_path (str): The path to the folder containing `.jsonl` files.
            save_path (str): The path to save the merged `.jsonl` file.
        """
        try:
            async with aiofiles.open(save_path, mode='w') as save_file:
                for root, _, files in os.walk(folder_path):
                    for file in files:
                        if file.endswith('.jsonl'):
                            file_path = os.path.join(root, file)
                            async with aiofiles.open(file_path, mode='r') as jsonl_file:
                                async for line in jsonl_file:
                                    await save_file.write(line)
            print(f"Successfully merged files into {save_path}")
        except Exception as e:
            print(f"An error occurred: {e}")