#!/usr/bin/env python3
"""
Medical Terminology MCP Server
Model Context Protocol server for medical terminology analysis
"""

import asyncio
import json
import os
from typing import Any, Dict, List, Optional
from mcp.server import Server
from mcp.server.models import InitializationOptions
from mcp.server.stdio import stdio_server
from mcp.types import (
    CallToolRequest,
    ListToolsRequest,
    Tool,
    TextContent,
    ImageContent,
    EmbeddedResource,
)

# Import our terminology analyzer
from medical_terminology_analyzer import MedicalTerminologyAnalyzer

class TerminologyMCPServer:
    """MCP Server for Medical Terminology Analysis"""
    
    def __init__(self):
        self.server = Server("medical-terminology-analyzer")
        self.analyzer = None
        self.setup_handlers()
    
    def setup_handlers(self):
        """Setup MCP server handlers"""
        
        @self.server.list_tools()
        async def list_tools() -> List[Tool]:
            """List available terminology analysis tools"""
            return [
                Tool(
                    name="terminology_analyze",
                    description="Analyze medical terminology in text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Medical text to analyze"
                            },
                            "case_id": {
                                "type": "string",
                                "description": "Case identifier",
                                "default": "unknown"
                            }
                        },
                        "required": ["text"]
                    }
                ),
                Tool(
                    name="terminology_search",
                    description="Search for similar medical terms",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query for medical terms"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results",
                                "default": 10
                            }
                        },
                        "required": ["query"]
                    }
                ),
                Tool(
                    name="terminology_stats",
                    description="Get terminology analysis statistics",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="terminology_categories",
                    description="Get medical terminology categories",
                    inputSchema={
                        "type": "object",
                        "properties": {}
                    }
                ),
                Tool(
                    name="terminology_embedding",
                    description="Get vector embedding for medical text",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "text": {
                                "type": "string",
                                "description": "Text to get embedding for"
                            }
                        },
                        "required": ["text"]
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: Dict[str, Any]) -> List[TextContent]:
            """Handle tool calls"""
            
            if not self.analyzer:
                return [TextContent(
                    type="text",
                    text="❌ Terminology analyzer not initialized. Please check your Neon connection."
                )]
            
            try:
                if name == "terminology_analyze":
                    return await self.handle_analyze(arguments)
                elif name == "terminology_search":
                    return await self.handle_search(arguments)
                elif name == "terminology_stats":
                    return await self.handle_stats(arguments)
                elif name == "terminology_categories":
                    return await self.handle_categories(arguments)
                elif name == "terminology_embedding":
                    return await self.handle_embedding(arguments)
                else:
                    return [TextContent(
                        type="text",
                        text=f"❌ Unknown tool: {name}"
                    )]
            except Exception as e:
                return [TextContent(
                    type="text",
                    text=f"❌ Error executing {name}: {str(e)}"
                )]
    
    async def handle_analyze(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle terminology analysis"""
        text = arguments.get("text", "")
        case_id = arguments.get("case_id", "unknown")
        
        if not text:
            return [TextContent(
                type="text",
                text="❌ No text provided for analysis"
            )]
        
        # Analyze terminology
        analysis = self.analyzer.analyze_case_terminology(case_id, text)
        
        result = {
            "case_id": analysis.case_id,
            "total_terms": analysis.total_terms,
            "unique_terms": analysis.unique_terms,
            "categories": analysis.categories,
            "top_terms": analysis.top_terms,
            "medical_score": round(analysis.medical_score, 2),
            "complexity_score": round(analysis.complexity_score, 2)
        }
        
        return [TextContent(
            type="text",
            text=f"📊 **Medical Terminology Analysis**\n\n"
                 f"**Case ID:** {result['case_id']}\n"
                 f"**Total Terms:** {result['total_terms']}\n"
                 f"**Unique Terms:** {result['unique_terms']}\n"
                 f"**Medical Score:** {result['medical_score']}%\n"
                 f"**Complexity Score:** {result['complexity_score']}\n\n"
                 f"**Categories:** {json.dumps(result['categories'], indent=2)}\n\n"
                 f"**Top Terms:** {json.dumps(result['top_terms'], indent=2)}"
        )]
    
    async def handle_search(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle terminology search"""
        query = arguments.get("query", "")
        limit = arguments.get("limit", 10)
        
        if not query:
            return [TextContent(
                type="text",
                text="❌ No search query provided"
            )]
        
        # Get embedding for query
        embedding = self.analyzer.get_embedding(query)
        if not embedding:
            return [TextContent(
                type="text",
                text="❌ Failed to get embedding for search query"
            )]
        
        # Find similar terms
        similar = self.analyzer.neon_db.find_similar_terms(embedding, limit=limit)
        
        if not similar:
            return [TextContent(
                type="text",
                text="❌ No similar terms found"
            )]
        
        result_text = f"🔍 **Similar Medical Terms for '{query}'**\n\n"
        for i, term in enumerate(similar, 1):
            result_text += f"{i}. **{term['term']}** (Category: {term['category']})\n"
            result_text += f"   Frequency: {term['frequency']}, Similarity: {term['similarity']:.3f}\n\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def handle_stats(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle terminology statistics"""
        stats = self.analyzer.neon_db.get_terminology_stats()
        
        if not stats:
            return [TextContent(
                type="text",
                text="❌ Failed to get terminology statistics"
            )]
        
        result_text = f"📈 **Medical Terminology Statistics**\n\n"
        result_text += f"**Total Terms:** {stats.get('total_terms', 0)}\n\n"
        
        if 'categories' in stats:
            result_text += "**Category Distribution:**\n"
            for category, count in stats['categories'].items():
                result_text += f"- {category}: {count}\n"
            result_text += "\n"
        
        if 'top_terms' in stats:
            result_text += "**Most Frequent Terms:**\n"
            for i, (term, frequency) in enumerate(stats['top_terms'][:10], 1):
                result_text += f"{i}. {term}: {frequency}\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def handle_categories(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle terminology categories"""
        categories = self.analyzer.categories
        
        result_text = "🏥 **Medical Terminology Categories**\n\n"
        for category, keywords in categories.items():
            result_text += f"**{category.title()}:**\n"
            result_text += f"Keywords: {', '.join(keywords[:5])}{'...' if len(keywords) > 5 else ''}\n\n"
        
        return [TextContent(type="text", text=result_text)]
    
    async def handle_embedding(self, arguments: Dict[str, Any]) -> List[TextContent]:
        """Handle embedding generation"""
        text = arguments.get("text", "")
        
        if not text:
            return [TextContent(
                type="text",
                text="❌ No text provided for embedding"
            )]
        
        embedding = self.analyzer.get_embedding(text)
        if not embedding:
            return [TextContent(
                type="text",
                text="❌ Failed to generate embedding"
            )]
        
        return [TextContent(
            type="text",
            text=f"🔢 **Vector Embedding Generated**\n\n"
                 f"**Text:** {text}\n"
                 f"**Embedding Dimension:** {len(embedding)}\n"
                 f"**First 5 values:** {embedding[:5]}\n"
                 f"**Last 5 values:** {embedding[-5:]}"
        )]
    
    async def initialize_analyzer(self):
        """Initialize the terminology analyzer"""
        try:
            # Load environment
            from dotenv import load_dotenv
            load_dotenv()
            
            neon_connection = os.getenv('NEON_CONNECTION_STRING')
            openai_key = os.getenv('OPENAI_API_KEY')
            
            if not neon_connection or not openai_key:
                print("❌ Missing environment variables. Please check your .env file.")
                return False
            
            # Initialize analyzer
            self.analyzer = MedicalTerminologyAnalyzer(neon_connection, openai_key)
            
            # Connect to database
            if not self.analyzer.neon_db.connect():
                print("❌ Failed to connect to Neon database")
                return False
            
            print("✅ Medical Terminology MCP Server initialized")
            return True
            
        except Exception as e:
            print(f"❌ Failed to initialize analyzer: {e}")
            return False
    
    async def run(self):
        """Run the MCP server"""
        # Initialize analyzer
        await self.initialize_analyzer()
        
        # Run the server
        async with stdio_server() as (read_stream, write_stream):
            await self.server.run(
                read_stream,
                write_stream,
                InitializationOptions(
                    server_name="medical-terminology-analyzer",
                    server_version="1.0.0",
                    capabilities=self.server.get_capabilities(
                        notification_options=None,
                        experimental_capabilities=None,
                    ),
                ),
            )

async def main():
    """Main function"""
    server = TerminologyMCPServer()
    await server.run()

if __name__ == "__main__":
    asyncio.run(main())




