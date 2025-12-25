import logging
import os
import requests
import asyncio
from datetime import datetime
from livekit.agents import function_tool
from chromadb import PersistentClient
from sentence_transformers import SentenceTransformer

logger = logging.getLogger("tools")

# Initialize embedding model and ChromaDB
_embedding_model = None
_chroma_client = None
_collection = None


def get_embedding_model():
    """Lazy load embedding model"""
    global _embedding_model
    if _embedding_model is None:
        logger.info("Loading embedding model...")
        _embedding_model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')
    return _embedding_model


def get_chroma_collection():
    """Lazy load ChromaDB collection"""
    global _chroma_client, _collection
    if _collection is None:
        logger.info("Loading ChromaDB collection...")
        _chroma_client = PersistentClient(path="./chroma_db")
        _collection = _chroma_client.get_collection(name="arabica_qa")
    return _collection


@function_tool
async def search_knowledge_base(query: str) -> str:
    """
    Search through Arabic knowledge base for factual information.
    
    Use this tool when:
    - User asks factual questions about history, science, geography, culture
    - Questions require specific factual knowledge
    - Questions are in Arabic or about Arabic topics
    
    Do NOT use for:
    - Current events or recent news (use search_web instead)
    - Weather information (use get_current_weather instead)
    - Simple greetings or general conversation
    
    Args:
        query: The search query in Arabic or English
        
    Returns:
        Relevant information from the knowledge base
        
    Example queries:
    - "ما هي عاصمة مصر؟"
    - "من هو أول رئيس للمغرب؟"
    - "أخبرني عن تاريخ الأندلس"
    """
    logger.info(f"🔍 RAG Tool called: {query}")
    
    try:
        # Get models/collections
        model = get_embedding_model()
        collection = get_chroma_collection()
        
        # Generate query embedding
        query_embedding = model.encode(query).tolist()
        
        # Search ChromaDB
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=3
        )
        
        # Format results
        if results['documents'] and results['documents'][0]:
            contexts = results['documents'][0]
            formatted = "المعلومات من قاعدة المعرفة:\n\n"
            for i, context in enumerate(contexts, 1):
                formatted += f"{i}. {context[:300]}...\n\n"
            
            logger.info(f"✅ Found {len(contexts)} relevant documents")
            await asyncio.sleep(0.1)
            return formatted
        else:
            logger.info("No results found in knowledge base")
            return "لم أجد معلومات ذات صلة في قاعدة المعرفة."
            
    except Exception as e:
        logger.error(f"❌ RAG Error: {e}")
        return "عذراً، حدث خطأ أثناء البحث في قاعدة المعرفة."


@function_tool
async def get_current_weather(city: str) -> str:  
    """
    Tool for current weather, recent weather, and forecasts
    using WeatherAPI.com

    Use this tool when:
    - User asks about current weather, temperature, or conditions
    - Questions contain words like: طقس، حرارة، أمطار، جو
    - User wants to know "how's the weather in [city]"
    
    Do NOT use for:
    - Weather forecasts (not supported yet)
    - Historical weather data
    - General information about cities
    
    Args:
        city: Name of the city (WARNING: Recomended to use English City Names)
              Examples: "الرباط" -> "Rabat", "الدار البيضاء" -> "Casablanca"
        
    Returns:
        Current weather conditions including temperature, humidity, and conditions
        
    Example queries:
    - "ما هو الطقس في الرباط؟"
    - "What's the weather like in Casablanca?"
    - "كيف الجو اليوم في بني ملال؟"
    """

    logger.info(f"🌤️ Weather tool called for: {city}")
    
    try:
        api_key = os.getenv("WEATHERAPI_KEY")
        url = f"http://api.weatherapi.com/v1/current.json?key={api_key}&q={city}&aqi=no"
        
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        current = data["current"]
        location = data["location"]
        
        # Return formatted STRING instead of dict
        result = f"""الطقس في {location['name']}, {location['country']}:
- درجة الحرارة: {current['temp_c']}°C
- الإحساس بالحرارة: {current['feelslike_c']}°C
- الرطوبة: {current['humidity']}%
- الحالة: {current['condition']['text']}
- سرعة الرياح: {current['wind_kph']} كم/ساعة"""
        
        logger.info(f"✅ Weather retrieved: {current['temp_c']}°C, {current['condition']['text']}")
        return result
        
    except Exception as e:
        logger.error(f"❌ Weather API Error: {e}")
        return f"عذراً، لم أتمكن من الحصول على الطقس لـ {city}"


@function_tool
async def search_web(query: str) -> str:
    """
    Search the web for current information, news, and recent events.
    
    Use this tool when:
    - User asks about recent news or current events
    - Questions require up-to-date information (after January 2025)
    - User explicitly asks to "search" or "look up" something
    - Questions contain words like: آخر الأخبار، ابحث، جديد
    
    Do NOT use for:
    - Well-known historical facts (answer directly)
    - Weather information (use get_current_weather)
    - General knowledge questions
    
    Args:
        query: Search query in any language
        
    Returns:
        Recent search results with titles, summaries, and sources
        
    Example queries:
    - "ابحث عن آخر أخبار المغرب"
    - "What are the latest AI developments?"
    - "ما الجديد في عالم التكنولوجيا؟"
    """
    logger.info(f"🔎 Web search called: {query}")
    
    try:
        api_key = os.getenv("TAVILY_API_KEY")
        url = "https://api.tavily.com/search"
        
        payload = {
            "api_key": api_key,
            "query": query,
            "search_depth": "basic",
            "max_results": 3
        }
        
        response = requests.post(url, json=payload, timeout=8)
        response.raise_for_status()
        data = response.json()
        
        if data.get("results"):
            formatted = "نتائج البحث:\n\n"
            for i, result in enumerate(data["results"][:3], 1):
                formatted += f"{i}. {result['title']}\n"
                formatted += f"   {result['content'][:200]}...\n"
                formatted += f"   المصدر: {result['url']}\n\n"
            
            logger.info(f"✅ Found {len(data['results'])} results")
            await asyncio.sleep(0.1)
            return formatted
        else:
            return "لم أجد نتائج ذات صلة بهذا الموضوع."
            
    except Exception as e:
        logger.error(f"❌ Web Search Error: {e}")
        return "عذراً، حدث خطأ أثناء البحث على الإنترنت."