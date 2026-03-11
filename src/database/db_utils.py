import os
from neo4j import GraphDatabase, AsyncGraphDatabase
from litellm import acompletion
from dotenv import load_dotenv

load_dotenv()

class Neo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    def close(self):
        self.driver.close()

class AsyncNeo4jClient:
    def __init__(self, uri=None, user=None, password=None):
        self.uri = uri or os.getenv("NEO4J_URI", "bolt://localhost:7687")
        self.user = user or os.getenv("NEO4J_USER", "neo4j")
        self.password = password or os.getenv("NEO4J_PASSWORD", "password")
        self.driver = AsyncGraphDatabase.driver(self.uri, auth=(self.user, self.password))
        
    async def close(self):
        await self.driver.close()

class AsyncLiteLLMClient:
    def __init__(self, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.model_name = model_name
        
    async def generate_response(self, system_prompt: str, user_message: str, temperature: float = 0.1) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message}
        ]
        
        try:
            response = await acompletion(
                model=self.model_name,
                messages=messages,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error during async LLM call: {e}")
            raise
