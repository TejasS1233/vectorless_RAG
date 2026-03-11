import json
import asyncio
import re
from src.database.db_utils import AsyncNeo4jClient, AsyncLiteLLMClient

EXTRACTOR_PROMPT = """You are an advanced knowledge graph extraction agent.
Your task is to identify if the given Target Section semantically references or heavily relies on any OTHER sections anywhere in the database.

You will be provided with:
1. The Target Section (the text you are analyzing)
2. A List of Available Sections (all other sections across all documents)

If the Target Section explicitly references concepts, chapters, or topics that are the main focus of another Available Section, extract that connection. 
Only return a JSON list containing the exact "Composite IDs" of the referenced sections.
Do NOT hallucinate connections. If there are no clear, explicit references, return an empty list: []

Example output format:
["earthmover.pdf::0004", "other_doc.pdf::0012"]
"""

async def build_cross_references(doc_name: str = None, model_name: str = "groq/llama-3.3-70b-versatile"):
    print(f"\n[CROSS-DOCUMENT REFERENCE EXTRACTION] Starting semantic cross-reference extraction...")
    neo4j_client = AsyncNeo4jClient()
    llm_client = AsyncLiteLLMClient(model_name=model_name)

    # Fetch ALL sections in the DB
    query_all = """
    MATCH (s:Section)
    RETURN s.doc_name AS doc_name, s.node_id AS id, s.title AS title, s.summary AS summary, s.text AS text
    """
    
    try:
        async with neo4j_client.driver.session() as session:
            result = await session.run(query_all)
            all_sections = await result.data()
            
        if not all_sections:
            print("[REFERENCE EXTRACTION] No sections found in database.")
            return

        print(f"[REFERENCE EXTRACTION] Found {len(all_sections)} total sections in the database. Building connections utilizing LLM reasoning...")

        # Generate lightweight context string listing all valid targets across all documents
        available_sections_context = "AVAILABLE SECTIONS ACROSS ALL DOCUMENTS:\n"
        for s in all_sections:
            composite_id = f"{s['doc_name']}::{s['id']}"
            available_sections_context += f"Composite ID: {composite_id} | Title: {s['title']}\n"

        links_created = 0
        
        # If doc_name is provided, we only analyze sections FROM that document as sources.
        # Otherwise, we analyze every section in the database as a source (heavy).
        source_sections = [s for s in all_sections if s['doc_name'] == doc_name] if doc_name else all_sections
        
        for index, current_section in enumerate(source_sections):
            print(f"  Analyzing ({index+1}/{len(source_sections)}): '{current_section['title']}' (from {current_section['doc_name']})...")
            
            # Truncate text slightly to save context window if it is massive
            body_text_excerpt = current_section['text']
            if body_text_excerpt and len(body_text_excerpt) > 3000:
                body_text_excerpt = body_text_excerpt[:3000] + "\n...[truncated for length]"
                
            composite_id = f"{current_section['doc_name']}::{current_section['id']}"
            target_context = f"TARGET SECTION:\nComposite ID: {composite_id}\nTitle: {current_section['title']}\nSummary: {current_section['summary']}\nText Excerpt: {body_text_excerpt}"
            
            user_message = f"{available_sections_context}\n\n{target_context}\n\nReturn ONLY a JSON list of referenced Composite IDs:"
            
            response_text = await llm_client.generate_response(EXTRACTOR_PROMPT, user_message)
            
            # Clean and parse JSON strictly
            cleaned_text = re.sub(r'```(?:json)?\n?(.*?)\n?```', r'\1', response_text, flags=re.DOTALL)
            start = cleaned_text.find('[')
            end = cleaned_text.rfind(']') + 1
            
            referenced_ids = []
            if start != -1 and end != 0:
                try:
                    referenced_ids = json.loads(cleaned_text[start:end])
                except json.JSONDecodeError:
                    pass
                    
            valid_ids = []
            for ref in referenced_ids:
                if ref != composite_id and any(f"{s['doc_name']}::{s['id']}" == ref for s in all_sections):
                    valid_ids.append(ref)
            
            if valid_ids:
                print(f"    -> Found semantic references pointing to: {valid_ids}")
                async with neo4j_client.driver.session() as session:
                    for target_comp_id in valid_ids:
                        t_doc, t_id = target_comp_id.split("::")
                        edge_query = """
                        MATCH (s:Section {node_id: $source_id, doc_name: $s_doc})
                        MATCH (t:Section {node_id: $target_id, doc_name: $t_doc})
                        MERGE (s)-[:REFERENCES]->(t)
                        """
                        await session.run(edge_query, 
                            source_id=current_section['id'], s_doc=current_section['doc_name'],
                            target_id=t_id, t_doc=t_doc)
                        links_created += 1

        print(f"\n[REFERENCE EXTRACTION] Completed! Successfully injected {links_created} new semantic cross-references into the knowledge graph.")
        
    except Exception as e:
         print(f"[REFERENCE EXTRACTION] Fatal Error: {e}")
    finally:
        await neo4j_client.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Extract cross-references across all documents inside the Neo4j database")
    parser.add_argument("--doc_name", type=str, required=False, help="If provided, only analyzes this document as a source (but can link to any other docs).")
    args = parser.parse_args()
    asyncio.run(build_cross_references(args.doc_name))
