import argparse
import asyncio
import json
from src.database.db_utils import AsyncNeo4jClient, AsyncLiteLLMClient

SYSTEM_PROMPT = """You are an intelligent knowledge agent designed to reason over a document's hierarchical structure.
You will be provided with a user query and a list of document sections (nodes in a graph).
For each section, you will see its ID, Title, Summary (if available), and Page/Text Index.

Your goal is to determine which sections MIGHT contain the answer to the user's query.

Respond ONLY with a JSON list of node IDs that are most relevant. 
If none are relevant, return an empty list: []

Example response:
["0001", "0005"]
"""

class Neo4jRetriever:
    def __init__(self, doc_name: str, model_name: str = "groq/llama-3.3-70b-versatile"):
        self.doc_name = doc_name
        self.neo4j_client = AsyncNeo4jClient()
        self.llm_client = AsyncLiteLLMClient(model_name=model_name)

    async def get_root_sections(self):
        query = """
        MATCH (d:Document {name: $doc_name})-[:HAS_SECTION]->(s:Section)
        RETURN s.node_id AS id, s.title AS title, s.summary AS summary, 
               s.start_index AS start_index, s.end_index AS end_index
        ORDER BY s.start_index
        """
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(query, doc_name=self.doc_name)
            records = await result.data()
            return records

    async def get_sub_sections(self, parent_id: str):
        query = """
        MATCH (p:Section {node_id: $parent_id, doc_name: $doc_name})-[:HAS_SUBSECTION]->(s:Section)
        RETURN s.node_id AS id, s.title AS title, s.summary AS summary, 
               s.start_index AS start_index, s.end_index AS end_index
        ORDER BY s.start_index
        """
        async with self.neo4j_client.driver.session() as session:
            result = await session.run(query, parent_id=parent_id, doc_name=self.doc_name)
            records = await result.data()
            return records

    async def ask_llm_for_relevant_nodes(self, query: str, sections: list) -> list:
        if not sections:
            return []
            
        sections_text = ""
        for sec in sections:
            sections_text += f"\n- ID: {sec['id']} | Title: {sec['title']}"
            if sec['summary']:
                sections_text += f" | Summary: {sec['summary']}"
                
        user_message = f"User Query: {query}\n\nAvailable Sections:{sections_text}\n\nReturn JSON list of relevant IDs:"
        
        try:
            response_text = await self.llm_client.generate_response(SYSTEM_PROMPT, user_message)
            print(f"Raw Output: {response_text}")
            import re
            
            cleaned_text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL)
            
            start = cleaned_text.find('[')
            end = cleaned_text.rfind(']') + 1
            if start != -1 and end != 0:
                json_str = cleaned_text[start:end]
                return json.loads(json_str)
            else:
                print(f"Could not parse JSON from LLM.")
                return []
        except Exception as e:
            print(f"Error reasoning over sections: {e}")
            return []

    async def retrieve(self, query: str):
        print(f"Searching for answers to: '{query}' in document: {self.doc_name}")
        
        current_layer = await self.get_root_sections()
        if not current_layer:
            print(f"No root sections found for {self.doc_name}. Is it ingested?")
            return []

        relevant_leaf_nodes = []
        
        queue = [current_layer]
        
        while queue:
            sections_to_check = queue.pop(0)
            
            relevant_ids = await self.ask_llm_for_relevant_nodes(query, sections_to_check)
            print(f"LLM highlighted IDs: {relevant_ids} among {[s['id'] for s in sections_to_check]}")
            
            for rid in relevant_ids:
                node_data = next((s for s in sections_to_check if s['id'] == rid), None)
                if not node_data:
                    continue
                    
                children = await self.get_sub_sections(rid)
                
                if children:
                    print(f"Drilling down into '{node_data['title']}'...")
                    queue.append(children)
                else:
                    print(f"Found relevant leaf node: '{node_data['title']}' (Pages {node_data['start_index']}-{node_data['end_index']})")
                    relevant_leaf_nodes.append(node_data)
                    
        return relevant_leaf_nodes
        
    async def close(self):
        await self.neo4j_client.close()

async def main():
    parser = argparse.ArgumentParser(description="Agentic Retrieval over Neo4j structure")
    parser.add_argument("--doc_name", type=str, required=True, help="Name of the ingested document")
    parser.add_argument("--query", type=str, required=True, help="User query to search for")
    parser.add_argument("--model", type=str, default="groq/llama-3.3-70b-versatile", help="LiteLLM model name")
    args = parser.parse_args()
    
    retriever = Neo4jRetriever(doc_name=args.doc_name, model_name=args.model)
    try:
        results = await retriever.retrieve(args.query)
        print("\n=== FINAL RETRIEVAL RESULTS ===")
        if not results:
            print("No relevant sections found.")
        for r in results:
            print(f"- Title: {r['title']}")
            print(f"  Pages: {r['start_index']} to {r['end_index']}")
            print(f"  ID: {r['id']}")
    finally:
        await retriever.close()

if __name__ == "__main__":
    asyncio.run(main())
