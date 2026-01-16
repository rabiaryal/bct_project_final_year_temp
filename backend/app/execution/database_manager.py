"""
Database Connection Manager for Execution System
Handles MongoDB and FAISS connections with optimizations for location and fee queries
"""

# from pymongo import MongoClient

# # Requires the PyMongo package.
# # https://api.mongodb.com/python/current

# client = MongoClient('mongodb://ac-dndkjet-shard-00-02.bnyhztw.mongodb.net,ac-dndkjet-shard-00-01.bnyhztw.mongodb.net,ac-dndkjet-shard-00-00.bnyhztw.mongodb.net/?tls=true&authMechanism=MONGODB-X509&authSource=%24external&serverMonitoringMode=poll&maxIdleTimeMS=30000&minPoolSize=0&maxPoolSize=5&maxConnecting=6&replicaSet=atlas-4sixtq-shard-0&appName=Data+Explorer--695a442353349040d2343af3')
# filter={}

# result = client['crs']['college data'].find(
#   filter=filter
# )

from typing import Optional, Dict, Any, List
import asyncio
import logging
from datetime import datetime

import pymongo
from pymongo import MongoClient
from motor.motor_asyncio import AsyncIOMotorClient, AsyncIOMotorDatabase, AsyncIOMotorCollection

from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionDatabaseManager:
    """
    Database connection manager specifically optimized for execution queries
    Handles both MongoDB and FAISS connections with performance optimizations
    """
    
    def __init__(self):
        self.mongo_client: Optional[AsyncIOMotorClient] = None
        self.database: Optional[AsyncIOMotorDatabase] = None
        self.colleges_collection: Optional[AsyncIOMotorCollection] = None
        self.connection_status = {
            "mongodb": False,
            "indexes_created": False,
            "last_health_check": None
        }
    
    async def initialize_connections(self) -> bool:
        """Initialize database connections with optimizations"""
        try:
            logger.info("🔌 Initializing execution database connections...")
            
            # Initialize MongoDB
            await self._connect_mongodb()
            
            # Create optimized indexes
            await self._create_execution_indexes()
            
            # Health check
            await self._verify_connections()
            
            logger.info("✅ Execution database connections initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize execution database connections: {e}")
            return False
    
    async def _connect_mongodb(self):
        """Connect to MongoDB with execution-optimized settings"""
        try:
            # Use centralized config for MongoDB connection
            from ..utils.config import config
            
            # MongoDB connection string
            mongodb_uri = config.database.mongodb_uri
            
            # Mask credentials for logging
            masked_uri = mongodb_uri.split('@')[-1] if '@' in mongodb_uri else 'localhost'
            logger.info(f"🔌 Connecting to MongoDB at: {masked_uri}")
            
            self.mongo_client = AsyncIOMotorClient(
                mongodb_uri,
                maxPoolSize=20,
                minPoolSize=5,
                maxIdleTimeMS=30000,
                waitQueueTimeoutMS=5000,
                serverSelectionTimeoutMS=5000,
                connectTimeoutMS=5000
            )
            
            # Select database and collection based on config
            self.database = self.mongo_client[config.database.database_name]
            self.colleges_collection = self.database[config.database.collection_name]
            
            # Test connection
            await self.mongo_client.admin.command('ping')
            self.connection_status["mongodb"] = True
            
            logger.info("📊 MongoDB connection established")
            
        except Exception as e:
            logger.error(f"❌ MongoDB connection failed: {e}")
            raise
    
    async def _create_execution_indexes(self):
        """Create optimized indexes for location and fee queries"""
        try:
            if self.colleges_collection is None:
                return
            
            # Create indexes for common execution queries
            indexes_to_create = [
                # Location index for faster location filtering
                ("Location", pymongo.TEXT),
                
                # Compound index for location + fee queries
                [("Location", pymongo.TEXT), ("Fees", pymongo.ASCENDING)],
                
                # Name index for identifier lookups
                ("Name", pymongo.TEXT),
                
                # Type index for filtering by college type
                ("Type", pymongo.ASCENDING),
                
                # Compound index for semantic search with filters
                [("Name", pymongo.TEXT), ("Location", pymongo.TEXT), ("Type", pymongo.ASCENDING)]
            ]
            
            for index_spec in indexes_to_create:
                try:
                    if isinstance(index_spec, list):
                        await self.colleges_collection.create_index(index_spec)
                        logger.debug(f"Created compound index: {index_spec}")
                    else:
                        await self.colleges_collection.create_index(index_spec)
                        logger.debug(f"Created index: {index_spec}")
                except Exception as idx_error:
                    # Index might already exist, log but don't fail
                    logger.debug(f"Index creation warning: {idx_error}")
            
            self.connection_status["indexes_created"] = True
            logger.info("🚀 Execution-optimized indexes created")
            
        except Exception as e:
            logger.error(f"Index creation error: {e}")
            # Don't raise - indexes are optimization, not critical
    
    async def _verify_connections(self):
        """Verify all database connections are working"""
        try:
            # Test MongoDB
            if self.colleges_collection is not None:
                college_count = await self.colleges_collection.count_documents({})
                logger.info(f"📊 MongoDB: {college_count} colleges available")
            
            # Update health check timestamp
            self.connection_status["last_health_check"] = datetime.now().isoformat()
            
        except Exception as e:
            logger.error(f"❌ Connection verification failed: {e}")
            raise
    
    async def execute_location_query(self, location_filter: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Execute optimized location-based query
        Specifically designed for fast location filtering
        """
        try:
            if self.colleges_collection is None:
                logger.warning("MongoDB collection not available")
                return []
            
            logger.info(f"📍 Executing location query: {location_filter}")
            
            # Use text index for efficient location search
            pipeline = []
            
            # Match stage with location filter
            if location_filter:
                pipeline.append({"$match": location_filter})
            
            # Add location scoring for relevance
            if "Location" in location_filter and "$text" in str(location_filter):
                pipeline.extend([
                    {"$addFields": {"score": {"$meta": "textScore"}}},
                    {"$sort": {"score": {"$meta": "textScore"}}}
                ])
            else:
                # Default sort by name for consistent results
                pipeline.append({"$sort": {"Name": 1}})
            
            # Limit results
            pipeline.append({"$limit": limit})
            
            # Execute aggregation
            cursor = self.colleges_collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"📋 Location query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Location query error: {e}")
            return []
    
    async def execute_fee_query(self, fee_filter: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """
        Execute optimized fee-based query
        Handles fee range filtering with numeric conversions
        """
        try:
            if self.colleges_collection is None:
                logger.warning("MongoDB collection not available")
                return []
            
            logger.info(f"💰 Executing fee query: {fee_filter}")
            
            # Build aggregation pipeline for fee queries
            pipeline = []
            
            # Add fee processing stage
            pipeline.append({
                "$addFields": {
                    "numeric_fees": {
                        "$toDouble": {
                            "$arrayElemAt": [
                                {
                                    "$regexFindAll": {
                                        "input": "$Fees",
                                        "regex": r"\d+",
                                        "options": "i"
                                    }
                                },
                                0
                            ]
                        }
                    }
                }
            })
            
            # Apply fee filter with numeric comparison
            if fee_filter:
                # Convert string filters to numeric filters
                numeric_filter = self._convert_fee_filter_to_numeric(fee_filter)
                if numeric_filter:
                    pipeline.append({"$match": numeric_filter})
            
            # Sort by fees (ascending)
            pipeline.extend([
                {"$sort": {"numeric_fees": 1}},
                {"$limit": limit}
            ])
            
            # Execute aggregation
            cursor = self.colleges_collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"💳 Fee query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Fee query error: {e}")
            return []
    
    async def execute_combined_location_fee_query(self, 
                                                location_filter: Dict[str, Any], 
                                                fee_filter: Dict[str, Any], 
                                                limit: int = 20) -> List[Dict[str, Any]]:
        """
        Execute optimized combined location and fee query
        This is the most commonly requested query type
        """
        try:
            if self.colleges_collection is None:
                logger.warning("MongoDB collection not available")
                return []
            
            logger.info(f"🎯 Executing combined location+fee query")
            logger.info(f"📍 Location: {location_filter}")
            logger.info(f"💰 Fee: {fee_filter}")
            
            # Build comprehensive aggregation pipeline
            pipeline = []
            
            # Stage 1: Add numeric fee field
            pipeline.append({
                "$addFields": {
                    "numeric_fees": {
                        "$toDouble": {
                            "$arrayElemAt": [
                                {
                                    "$regexFindAll": {
                                        "input": "$Fees",
                                        "regex": r"\d+",
                                        "options": "i"
                                    }
                                },
                                0
                            ]
                        }
                    }
                }
            })
            
            # Stage 2: Combine location and fee filters
            combined_match = {}
            
            # Add location filter
            if location_filter:
                combined_match.update(location_filter)
            
            # Add numeric fee filter
            if fee_filter:
                numeric_fee_filter = self._convert_fee_filter_to_numeric(fee_filter)
                if numeric_fee_filter:
                    combined_match.update(numeric_fee_filter)
            
            if combined_match:
                pipeline.append({"$match": combined_match})
            
            # Stage 3: Add relevance scoring
            pipeline.append({
                "$addFields": {
                    "relevance_score": {
                        "$add": [
                            # Location relevance
                            {"$cond": {
                                "if": {"$regexMatch": {"input": "$Location", "regex": "kathmandu", "options": "i"}},
                                "then": 0.3,
                                "else": 0.1
                            }},
                            # Fee relevance (lower fees score higher)
                            {"$cond": {
                                "if": {"$lt": ["$numeric_fees", 500000]},
                                "then": 0.2,
                                "else": 0.1
                            }}
                        ]
                    }
                }
            })
            
            # Stage 4: Sort by relevance score and fees
            pipeline.extend([
                {"$sort": {"relevance_score": -1, "numeric_fees": 1}},
                {"$limit": limit}
            ])
            
            # Execute aggregation
            cursor = self.colleges_collection.aggregate(pipeline)
            results = await cursor.to_list(length=limit)
            
            logger.info(f"🎯 Combined query returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Combined query error: {e}")
            return []
    
    async def execute_identifier_lookup(self, identifier_filter: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute identifier-based lookup (college name search)
        Optimized for exact and fuzzy name matching
        """
        try:
            if not self.colleges_collection:
                logger.warning("MongoDB collection not available")
                return []
            
            logger.info(f"🏫 Executing identifier lookup: {identifier_filter}")
            
            # Use text search for name matching
            pipeline = []
            
            # Add text search scoring
            if "Name" in identifier_filter:
                name_filter = identifier_filter["Name"]
                if isinstance(name_filter, dict) and "$text" in str(name_filter):
                    # Use text search with scoring
                    pipeline.extend([
                        {"$match": identifier_filter},
                        {"$addFields": {"score": {"$meta": "textScore"}}},
                        {"$sort": {"score": {"$meta": "textScore"}}},
                        {"$limit": 10}
                    ])
                else:
                    # Regular regex search
                    pipeline.extend([
                        {"$match": identifier_filter},
                        {"$sort": {"Name": 1}},
                        {"$limit": 10}
                    ])
            else:
                pipeline.extend([
                    {"$match": identifier_filter},
                    {"$sort": {"Name": 1}},
                    {"$limit": 10}
                ])
            
            # Execute query
            cursor = self.colleges_collection.aggregate(pipeline)
            results = await cursor.to_list(length=10)
            
            logger.info(f"🏫 Identifier lookup returned {len(results)} results")
            return results
            
        except Exception as e:
            logger.error(f"❌ Identifier lookup error: {e}")
            return []
    
    def _convert_fee_filter_to_numeric(self, fee_filter: Dict[str, Any]) -> Dict[str, Any]:
        """Convert string-based fee filters to numeric filters"""
        numeric_filter = {}
        
        for field, condition in fee_filter.items():
            if field in ["Fees", "fees", "fee"]:
                if isinstance(condition, dict):
                    numeric_condition = {}
                    for op, value in condition.items():
                        try:
                            # Convert string values to numeric
                            if isinstance(value, str):
                                numeric_value = float(value)
                            else:
                                numeric_value = value
                            
                            numeric_condition[op] = numeric_value
                        except (ValueError, TypeError):
                            # Skip invalid numeric conversions
                            continue
                    
                    if numeric_condition:
                        numeric_filter["numeric_fees"] = numeric_condition
        
        return numeric_filter
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics for monitoring"""
        try:
            stats = {
                "connection_status": self.connection_status.copy(),
                "mongodb": {
                    "connected": self.connection_status["mongodb"],
                    "college_count": 0
                }
            }
            
            if self.colleges_collection:
                stats["mongodb"]["college_count"] = await self.colleges_collection.count_documents({})
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {e}")
            return {"error": str(e)}
    
    async def close_connections(self):
        """Clean up database connections"""
        try:
            if self.mongo_client:
                self.mongo_client.close()
                logger.info("📊 MongoDB connections closed")
            
            self.connection_status = {
                "mongodb": False,
                "indexes_created": False,
                "last_health_check": None
            }
            
        except Exception as e:
            logger.error(f"Error closing connections: {e}")

# Global database manager instance
db_manager = ExecutionDatabaseManager()