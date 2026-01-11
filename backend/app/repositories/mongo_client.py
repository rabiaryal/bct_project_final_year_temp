"""MongoDB Repository"""

from motor.motor_asyncio import AsyncIOMotorClient
from pymongo.errors import ServerSelectionTimeoutError, ConfigurationError
from typing import Dict, List, Any, Optional
import asyncio
from datetime import datetime

from app.utils.logger import get_logger
from app.utils.config import config

logger = get_logger(__name__)

class MongoRepository:
    """MongoDB data access layer"""
    
    def __init__(self):
        self.client = None
        self.database = None
        self.collection = None
        self._connection_pool = None
        
    async def connect(self):
        """Establish MongoDB connection"""
        try:
            logger.info("Connecting to MongoDB Atlas...")
            
            # Create async MongoDB client with shorter timeouts
            self.client = AsyncIOMotorClient(
                config.database.mongodb_uri,
                serverSelectionTimeoutMS=5000,  # 5 seconds
                connectTimeoutMS=5000,  # 5 seconds
                maxPoolSize=10,
                minPoolSize=1
            )
            
            # Test connection with timeout
            await asyncio.wait_for(
                self.client.admin.command('ping'),
                timeout=5.0  # 5 second timeout
            )
            
            # Set database and collection
            self.database = self.client[config.database.database_name]
            self.collection = self.database[config.database.collection_name]
            
            # Get collection stats
            stats = await self.database.command("collStats", config.database.collection_name)
            logger.info(f"Connected successfully - Documents: {stats.get('count', 0)}")
            
        except asyncio.TimeoutError:
            logger.error("MongoDB connection timed out after 5 seconds")
            raise Exception("MongoDB connection timeout")
        except Exception as e:
            logger.error(f"MongoDB connection failed: {e}")
            raise
    
    async def disconnect(self):
        """Close MongoDB connection"""
        if self.client:
            self.client.close()
            logger.info("MongoDB connection closed")
    
    async def health_check(self) -> Dict[str, Any]:
        """Check MongoDB health"""
        try:
            if not self.client:
                return {"status": "disconnected", "error": "No client"}
                
            # Test connection with timeout
            await asyncio.wait_for(
                self.client.admin.command('ping'), 
                timeout=5.0
            )
            
            # Get basic stats
            stats = await self.database.command("collStats", config.database.collection_name)
            
            return {
                "status": "healthy",
                "document_count": stats.get('count', 0),
                "database": config.database.database_name,
                "collection": config.database.collection_name,
                "last_check": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def search_colleges(self, query_params: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Search colleges with hybrid approach"""
        try:
            # Build MongoDB query
            mongo_query = {}
            
            # College name search (exact + fuzzy)
            if college_name := query_params.get("college_name"):
                mongo_query["$or"] = [
                    {"name": {"$regex": college_name, "$options": "i"}},
                    {"name": {"$regex": college_name.replace(" ", ".*"), "$options": "i"}}
                ]
            
            # Location search
            if location := query_params.get("location"):
                if "$or" not in mongo_query:
                    mongo_query["$or"] = []
                mongo_query["$or"].extend([
                    {"location": {"$regex": location, "$options": "i"}},
                    {"address": {"$regex": location, "$options": "i"}}
                ])
            
            # Course search
            if course := query_params.get("course_name"):
                mongo_query["courses"] = {"$regex": course, "$options": "i"}
            
            # Facility search
            if facility := query_params.get("facility"):
                mongo_query["facilities"] = {"$regex": facility, "$options": "i"}
            
            # If no specific query, return limited results
            if not mongo_query:
                mongo_query = {}
            
            logger.debug(f"MongoDB query: {mongo_query}")
            
            # Execute query with limit
            cursor = self.collection.find(mongo_query).limit(10)
            results = await cursor.to_list(length=10)
            
            logger.info(f"Found {len(results)} colleges")
            return results
            
        except Exception as e:
            logger.error(f"College search error: {e}")
            return []
    
    async def find_college_by_name(self, college_name: str) -> Optional[Dict[str, Any]]:
        """Find specific college by name"""
        try:
            # Try exact match first
            college = await self.collection.find_one(
                {"name": {"$regex": f"^{college_name}$", "$options": "i"}}
            )
            
            # If no exact match, try fuzzy match
            if not college:
                college = await self.collection.find_one(
                    {"name": {"$regex": college_name, "$options": "i"}}
                )
            
            return college
            
        except Exception as e:
            logger.error(f"College lookup error: {e}")
            return None
    
    async def get_college_count(self) -> int:
        """Get total college count"""
        try:
            return await self.collection.count_documents({})
        except Exception as e:
            logger.error(f"Count error: {e}")
            return 0