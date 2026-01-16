"""
Execution Plan Handler
Executes database queries based on policy decisions with focus on location and fee criteria
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime

from ..repositories.mongo_client import MongoRepository
from ..services.college_retrieval_agent import CollegeRetrievalAgent
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionPlanHandler:
    """
    Handles execution of database queries based on policy decision execution plans
    Specifically optimized for location and fee filtering criteria
    """
    
    def __init__(self):
        self.mongo_repo = None
        self.faiss_agent = None
        self.initialized = False
        
    async def initialize(self):
        """Initialize database connections"""
        try:
            # Initialize MongoDB connection
            self.mongo_repo = MongoRepository()
            await self.mongo_repo.connect()
            
            # Initialize FAISS agent
            self.faiss_agent = CollegeRetrievalAgent()
            
            self.initialized = True
            logger.info("ExecutionPlanHandler initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize ExecutionPlanHandler: {e}")
            raise
    
    async def execute_plan(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute database queries based on policy decision execution plan
        
        Args:
            policy_decision: Complete policy decision with execution plan
            
        Returns:
            Execution results with database query results
        """
        if not self.initialized:
            await self.initialize()
        
        execution_plan = policy_decision.get("execution_plan")
        if not execution_plan:
            return self._build_no_execution_result(policy_decision)
        
        decision_action = policy_decision["decision"]["action"]
        
        if decision_action != "EXECUTE_QUERY":
            return self._build_non_query_result(policy_decision)
        
        logger.info(f"🚀 Executing query plan: {execution_plan['query_type']}")
        
        try:
            # Execute based on query type and strategy
            strategy = policy_decision["decision"]["strategy"]
            
            if strategy == "IDENTIFIER_LOOKUP":
                results = await self._execute_identifier_lookup(execution_plan)
            elif strategy == "FILTER_SEARCH":
                results = await self._execute_filter_search(execution_plan)
            elif strategy == "SEMANTIC_SEARCH":
                results = await self._execute_semantic_search(execution_plan)
            elif strategy == "RECOMMENDATION":
                results = await self._execute_recommendation(execution_plan)
            else:
                results = await self._execute_fallback_search(execution_plan)
            
            return self._build_execution_result(policy_decision, results)
            
        except Exception as e:
            logger.error(f"Error executing plan: {e}")
            return self._build_error_result(policy_decision, str(e))
    
    async def _execute_filter_search(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Execute filter search with focus on location and fee criteria
        Optimized for MongoDB structured queries with FAISS enhancement
        """
        logger.info("📊 Executing FILTER_SEARCH with location/fee criteria")
        
        results = []
        mongodb_results = []
        faiss_results = []
        
        # Step 1: MongoDB structured search (primary for filtering)
        if execution_plan["mongodb"]["enabled"]:
            mongodb_filter = execution_plan["mongodb"]["filter"]
            logger.info(f"🗄️ MongoDB filter: {mongodb_filter}")
            
            # Execute MongoDB query
            mongodb_results = await self._query_mongodb(mongodb_filter)
            logger.info(f"📋 MongoDB found {len(mongodb_results)} colleges")
        
        # Step 2: FAISS semantic search (enhancement)
        if execution_plan["faiss"]["enabled"] and len(mongodb_results) < 5:
            # Use FAISS if MongoDB results are limited
            faiss_query = execution_plan["faiss"]["query_text"]
            top_k = execution_plan["faiss"]["top_k"]
            
            logger.info(f"🔍 FAISS search: '{faiss_query}' (top_k={top_k})")
            faiss_results = await self._query_faiss(faiss_query, top_k)
            logger.info(f"🎯 FAISS found {len(faiss_results)} colleges")
        
        # Step 3: Combine and apply location/fee specific filtering
        results = await self._combine_and_filter_results(
            mongodb_results, 
            faiss_results,
            execution_plan["mongodb"]["filter"]
        )
        
        logger.info(f"✅ Filter search completed: {len(results)} final results")
        return results
    
    async def _execute_semantic_search(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute semantic search with FAISS primary, MongoDB enhancement"""
        logger.info("🔍 Executing SEMANTIC_SEARCH")
        
        results = []
        faiss_results = []
        mongodb_results = []
        
        # Step 1: FAISS semantic search (primary)
        if execution_plan["faiss"]["enabled"]:
            faiss_query = execution_plan["faiss"]["query_text"]
            top_k = execution_plan["faiss"]["top_k"]
            
            logger.info(f"🎯 FAISS semantic search: '{faiss_query}'")
            faiss_results = await self._query_faiss(faiss_query, top_k)
            logger.info(f"📋 FAISS found {len(faiss_results)} colleges")
        
        # Step 2: MongoDB enhancement (for additional structured data)
        if execution_plan["mongodb"]["enabled"]:
            mongodb_filter = execution_plan["mongodb"]["filter"]
            
            if mongodb_filter:  # Only if we have specific filters
                logger.info(f"🗄️ MongoDB enhancement filter: {mongodb_filter}")
                mongodb_results = await self._query_mongodb(mongodb_filter, limit=10)
                logger.info(f"📊 MongoDB enhancement found {len(mongodb_results)} colleges")
        
        # Step 3: Combine with semantic priority
        results = await self._combine_semantic_results(faiss_results, mongodb_results)
        
        logger.info(f"✅ Semantic search completed: {len(results)} final results")
        return results
    
    async def _execute_identifier_lookup(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute identifier lookup for specific college names"""
        logger.info("🏫 Executing IDENTIFIER_LOOKUP")
        
        results = []
        
        # For identifier lookup, prioritize exact matching
        mongodb_filter = execution_plan["mongodb"]["filter"]
        logger.info(f"🎯 Identifier MongoDB filter: {mongodb_filter}")
        
        # Step 1: MongoDB exact search
        mongodb_results = await self._query_mongodb(mongodb_filter, limit=5)
        logger.info(f"📋 MongoDB identifier search found {len(mongodb_results)} colleges")
        
        # Step 2: FAISS semantic backup if needed
        if len(mongodb_results) == 0 and execution_plan["faiss"]["enabled"]:
            faiss_query = execution_plan["faiss"]["query_text"]
            logger.info(f"🔍 FAISS backup search: '{faiss_query}'")
            faiss_results = await self._query_faiss(faiss_query, 3)
            
            # Convert FAISS results to standard format
            results = await self._enhance_faiss_with_mongodb(faiss_results)
        else:
            # Use MongoDB results
            results = await self._format_mongodb_results(mongodb_results)
        
        logger.info(f"✅ Identifier lookup completed: {len(results)} final results")
        return results
    
    async def _execute_recommendation(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute recommendation search with ranking"""
        logger.info("⭐ Executing RECOMMENDATION")
        
        # For recommendations, use FAISS for semantic similarity with ranking
        faiss_query = execution_plan["faiss"]["query_text"] or "best colleges"
        top_k = execution_plan["faiss"]["top_k"]
        
        faiss_results = await self._query_faiss(faiss_query, top_k)
        results = await self._enhance_faiss_with_mongodb(faiss_results)
        
        # Apply recommendation ranking
        results = self._apply_recommendation_ranking(results)
        
        logger.info(f"✅ Recommendation completed: {len(results)} final results")
        return results
    
    async def _execute_fallback_search(self, execution_plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Execute fallback search"""
        logger.info("🔄 Executing FALLBACK_SEARCH")
        
        # Simple MongoDB search without strict filters
        basic_filter = {}
        results_raw = await self._query_mongodb(basic_filter, limit=10)
        results = await self._format_mongodb_results(results_raw)
        
        logger.info(f"✅ Fallback search completed: {len(results)} final results")
        return results
    
    async def _query_mongodb(self, filter_conditions: Dict[str, Any], limit: int = 20) -> List[Dict[str, Any]]:
        """Query MongoDB with filter conditions"""
        try:
            if not self.mongo_repo:
                logger.warning("MongoDB not initialized, skipping query")
                return []
            
            # Apply location and fee specific optimizations
            optimized_filter = self._optimize_mongodb_filter(filter_conditions)
            
            colleges = await self.mongo_repo.find_colleges(optimized_filter, limit=limit)
            logger.debug(f"MongoDB query returned {len(colleges)} colleges")
            
            return colleges
            
        except Exception as e:
            logger.error(f"MongoDB query error: {e}")
            return []
    
    async def _query_faiss(self, query_text: str, top_k: int = 10) -> List[Dict[str, Any]]:
        """Query FAISS for semantic search"""
        try:
            if not self.faiss_agent:
                logger.warning("FAISS agent not initialized, skipping query")
                return []
            
            # Use the existing FAISS search from college retrieval agent
            # Create mock entities for the existing interface
            mock_entities = [{"entity": "GENERAL", "value": query_text, "confidence": 1.0}]
            
            search_results = await self.faiss_agent._execute_semantic_search(
                user_query=query_text,
                entities=mock_entities,
                filters={},
                top_n=top_k
            )
            
            logger.debug(f"FAISS query returned {len(search_results)} colleges")
            return search_results
            
        except Exception as e:
            logger.error(f"FAISS query error: {e}")
            return []
    
    def _optimize_mongodb_filter(self, filter_conditions: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize MongoDB filter for location and fee criteria"""
        optimized = filter_conditions.copy()
        
        # Location optimization - ensure case-insensitive search
        if "Location" in optimized:
            if isinstance(optimized["Location"], dict) and "$regex" in optimized["Location"]:
                optimized["Location"]["$options"] = "i"
        
        # Fee optimization - handle different fee formats
        if "Fees" in optimized or "fees" in optimized:
            fee_condition = optimized.get("Fees", optimized.get("fees"))
            if isinstance(fee_condition, dict):
                # Ensure numeric comparison for fees
                for op, value in fee_condition.items():
                    if op in ["$lt", "$lte", "$gt", "$gte"] and isinstance(value, str):
                        try:
                            optimized[list(optimized.keys())[0]][op] = float(value)
                        except ValueError:
                            pass
        
        logger.debug(f"Optimized MongoDB filter: {optimized}")
        return optimized
    
    async def _combine_and_filter_results(self, 
                                        mongodb_results: List[Dict], 
                                        faiss_results: List[Dict],
                                        filter_conditions: Dict[str, Any]) -> List[Dict]:
        """Combine MongoDB and FAISS results with location/fee filtering"""
        
        combined_results = []
        seen_colleges = set()
        
        # Prioritize MongoDB results (more structured for filtering)
        for college in mongodb_results:
            college_id = str(college.get("_id", college.get("name", "")))
            if college_id not in seen_colleges:
                formatted = await self._format_single_college_result(college, source="mongodb")
                if self._passes_location_fee_filter(formatted, filter_conditions):
                    combined_results.append(formatted)
                    seen_colleges.add(college_id)
        
        # Add FAISS results if we need more
        if len(combined_results) < 5:
            for college in faiss_results:
                college_name = college.get("name", "")
                if college_name not in seen_colleges:
                    if self._passes_location_fee_filter(college, filter_conditions):
                        combined_results.append(college)
                        seen_colleges.add(college_name)
        
        # Sort by confidence/relevance
        combined_results.sort(key=lambda x: x.get("confidence", 0.0), reverse=True)
        
        return combined_results[:10]  # Return top 10
    
    async def _combine_semantic_results(self, 
                                      faiss_results: List[Dict], 
                                      mongodb_results: List[Dict]) -> List[Dict]:
        """Combine semantic FAISS results with MongoDB enhancements"""
        
        # For semantic search, FAISS results take priority
        combined_results = list(faiss_results)  # Start with FAISS results
        
        # Add MongoDB results that aren't already in FAISS results
        faiss_names = {result.get("name", "").lower() for result in faiss_results}
        
        for college in mongodb_results:
            college_name = college.get("Name", college.get("name", "")).lower()
            if college_name not in faiss_names:
                formatted = await self._format_single_college_result(college, source="mongodb")
                combined_results.append(formatted)
        
        return combined_results[:10]
    
    def _passes_location_fee_filter(self, college: Dict[str, Any], filter_conditions: Dict[str, Any]) -> bool:
        """Check if college passes location and fee filters"""
        
        # Location filter
        if "Location" in filter_conditions:
            location_condition = filter_conditions["Location"]
            college_location = college.get("location", college.get("Location", "")).lower()
            
            if isinstance(location_condition, dict) and "$regex" in location_condition:
                required_location = location_condition["$regex"].lower()
                if required_location not in college_location:
                    return False
        
        # Fee filter
        fee_fields = ["Fees", "fees", "fee"]
        if any(field in filter_conditions for field in fee_fields):
            college_fees = None
            
            # Extract fee from college data
            for field in fee_fields:
                if field in college:
                    college_fees = college[field]
                    break
            
            if college_fees:
                try:
                    # Convert fee to numeric for comparison
                    if isinstance(college_fees, str):
                        import re
                        fee_numbers = re.findall(r'\d+', college_fees)
                        if fee_numbers:
                            college_fee_amount = float(fee_numbers[0])
                            
                            # Check against filter conditions
                            for field in fee_fields:
                                if field in filter_conditions:
                                    fee_condition = filter_conditions[field]
                                    if isinstance(fee_condition, dict):
                                        for op, value in fee_condition.items():
                                            if op == "$lt" and college_fee_amount >= value:
                                                return False
                                            elif op == "$gt" and college_fee_amount <= value:
                                                return False
                                            elif op == "$lte" and college_fee_amount > value:
                                                return False
                                            elif op == "$gte" and college_fee_amount < value:
                                                return False
                except (ValueError, TypeError):
                    pass  # Skip fee validation if conversion fails
        
        return True
    
    async def _format_mongodb_results(self, mongodb_results: List[Dict]) -> List[Dict]:
        """Format MongoDB results to standard format"""
        formatted_results = []
        
        for college in mongodb_results:
            formatted = await self._format_single_college_result(college, source="mongodb")
            formatted_results.append(formatted)
        
        return formatted_results
    
    async def _enhance_faiss_with_mongodb(self, faiss_results: List[Dict]) -> List[Dict]:
        """Enhance FAISS results with additional MongoDB data"""
        enhanced_results = []
        
        for faiss_result in faiss_results:
            # Try to get more detailed info from MongoDB
            college_name = faiss_result.get("name", "")
            
            if college_name:
                mongodb_filter = {"Name": {"$regex": college_name, "$options": "i"}}
                mongodb_colleges = await self._query_mongodb(mongodb_filter, limit=1)
                
                if mongodb_colleges:
                    # Merge FAISS confidence with MongoDB data
                    enhanced = await self._format_single_college_result(mongodb_colleges[0], source="faiss_enhanced")
                    enhanced["confidence"] = faiss_result.get("confidence", 0.0)
                    enhanced["faiss_score"] = faiss_result.get("confidence", 0.0)
                    enhanced_results.append(enhanced)
                else:
                    # Keep original FAISS result
                    enhanced_results.append(faiss_result)
            else:
                enhanced_results.append(faiss_result)
        
        return enhanced_results
    
    async def _format_single_college_result(self, college_data: Dict, source: str = "unknown") -> Dict[str, Any]:
        """Format a single college result to standard format"""
        
        return {
            "college_id": str(college_data.get("_id", college_data.get("college_id", "unknown"))),
            "name": college_data.get("Name", college_data.get("name", "Unknown College")),
            "location": college_data.get("Location", college_data.get("location", "Unknown Location")),
            "programs": college_data.get("Departments", college_data.get("programs", [])),
            "fees": college_data.get("Fees", college_data.get("fees", "Contact college")),
            "type": college_data.get("Type", college_data.get("type", "College")),
            "contact": college_data.get("ContactNumber", college_data.get("contact", "")),
            "website": college_data.get("Website", college_data.get("website", "")),
            "confidence": college_data.get("confidence", 0.8),
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
    
    def _apply_recommendation_ranking(self, results: List[Dict]) -> List[Dict]:
        """Apply recommendation-specific ranking to results"""
        
        # Simple ranking based on multiple factors
        for result in results:
            score = result.get("confidence", 0.0)
            
            # Boost score for certain criteria
            if "engineering" in result.get("type", "").lower():
                score += 0.1
            if "kathmandu" in result.get("location", "").lower():
                score += 0.05
            if result.get("programs") and len(result["programs"]) > 2:
                score += 0.05
            
            result["recommendation_score"] = score
        
        # Sort by recommendation score
        results.sort(key=lambda x: x.get("recommendation_score", 0.0), reverse=True)
        return results
    
    def _build_execution_result(self, policy_decision: Dict, results: List[Dict]) -> Dict[str, Any]:
        """Build execution result response"""
        return {
            "policy_id": policy_decision["policy_id"],
            "execution_status": "SUCCESS",
            "strategy": policy_decision["decision"]["strategy"],
            "results_count": len(results),
            "results": results,
            "execution_details": {
                "query_type": policy_decision["execution_plan"]["query_type"],
                "data_sources_used": policy_decision["execution_plan"]["data_sources"],
                "execution_timestamp": datetime.now().isoformat()
            }
        }
    
    def _build_no_execution_result(self, policy_decision: Dict) -> Dict[str, Any]:
        """Build result for policies with no execution plan"""
        return {
            "policy_id": policy_decision["policy_id"],
            "execution_status": "NO_EXECUTION_NEEDED",
            "action": policy_decision["decision"]["action"],
            "reason": policy_decision["decision"]["reason"],
            "results": []
        }
    
    def _build_non_query_result(self, policy_decision: Dict) -> Dict[str, Any]:
        """Build result for non-query actions"""
        return {
            "policy_id": policy_decision["policy_id"],
            "execution_status": "NON_QUERY_ACTION", 
            "action": policy_decision["decision"]["action"],
            "reason": policy_decision["decision"]["reason"],
            "results": []
        }
    
    def _build_error_result(self, policy_decision: Dict, error_message: str) -> Dict[str, Any]:
        """Build error result response"""
        return {
            "policy_id": policy_decision.get("policy_id", "unknown"),
            "execution_status": "ERROR",
            "error_message": error_message,
            "results": [],
            "execution_details": {
                "error_timestamp": datetime.now().isoformat()
            }
        }

# Global execution handler instance
execution_handler = ExecutionPlanHandler()