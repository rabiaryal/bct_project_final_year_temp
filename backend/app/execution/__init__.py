"""
Execution System Main Module
Orchestrates execution of database queries for location and fee criteria
"""

from typing import Dict, Any, List, Optional
import asyncio
import logging
from datetime import datetime

from .execution_plan_handler import execution_handler
from .location_fee_executor import location_fee_executor
from .database_manager import db_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class ExecutionSystem:
    """
    Main execution system that orchestrates database queries based on policy decisions
    Specialized for location and fee criteria with optimized performance
    """
    
    def __init__(self):
        self.execution_handler = execution_handler
        self.location_fee_executor = location_fee_executor
        self.db_manager = db_manager
        self.initialized = False
    
    async def initialize(self):
        """Initialize the entire execution system"""
        if not self.initialized:
            logger.info("🚀 Initializing Execution System...")
            
            try:
                # Initialize database manager
                await self.db_manager.initialize_connections()
                
                # Initialize execution handler
                await self.execution_handler.initialize()
                
                # Initialize specialized executor
                await self.location_fee_executor.initialize()
                
                self.initialized = True
                logger.info("✅ Execution System fully initialized")
                
            except Exception as e:
                logger.error(f"❌ Failed to initialize Execution System: {e}")
                raise
    
    async def execute_policy_decision(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main execution entry point - executes a policy decision
        Routes to appropriate specialized executors based on criteria type
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info(f"🎯 Executing policy decision: {policy_decision.get('policy_id', 'unknown')}")
        
        try:
            # Check if this is an execution-worthy decision
            if not self._should_execute(policy_decision):
                return self._build_non_execution_response(policy_decision)
            
            # Determine execution strategy based on criteria
            execution_strategy = self._determine_execution_strategy(policy_decision)
            
            logger.info(f"📊 Using execution strategy: {execution_strategy}")
            
            # Route to appropriate executor
            if execution_strategy == "LOCATION_ONLY":
                result = await self._execute_location_criteria(policy_decision)
            elif execution_strategy == "FEE_ONLY":
                result = await self._execute_fee_criteria(policy_decision)
            elif execution_strategy == "LOCATION_AND_FEE":
                result = await self._execute_combined_criteria(policy_decision)
            elif execution_strategy == "GENERAL":
                result = await self._execute_general_query(policy_decision)
            else:
                result = await self._execute_fallback(policy_decision)
            
            # Add execution metadata
            result["execution_metadata"] = {
                "strategy_used": execution_strategy,
                "policy_id": policy_decision.get("policy_id"),
                "execution_time": datetime.now().isoformat(),
                "system_version": "1.0.0"
            }
            
            logger.info(f"✅ Execution completed successfully with {result.get('results_count', 0)} results")
            return result
            
        except Exception as e:
            logger.error(f"❌ Execution failed: {e}")
            return self._build_error_response(policy_decision, str(e))
    
    def _should_execute(self, policy_decision: Dict[str, Any]) -> bool:
        """Determine if a policy decision requires database execution"""
        decision_action = policy_decision.get("decision", {}).get("action")
        
        # Only execute database queries
        if decision_action != "EXECUTE_QUERY":
            return False
        
        # Must have execution plan
        if not policy_decision.get("execution_plan"):
            return False
        
        return True
    
    def _determine_execution_strategy(self, policy_decision: Dict[str, Any]) -> str:
        """Determine the best execution strategy based on policy decision content"""
        
        execution_plan = policy_decision.get("execution_plan", {})
        mongodb_filter = execution_plan.get("mongodb", {}).get("filter", {})
        strategy = policy_decision.get("decision", {}).get("strategy", "")
        
        # Check for location and fee criteria
        has_location = any(key.lower() in ["location"] for key in mongodb_filter.keys())
        has_fee = any(key.lower() in ["fees", "fee"] for key in mongodb_filter.keys())
        
        # Determine strategy
        if has_location and has_fee:
            return "LOCATION_AND_FEE"
        elif has_location:
            return "LOCATION_ONLY"
        elif has_fee:
            return "FEE_ONLY"
        elif strategy == "IDENTIFIER_LOOKUP":
            return "GENERAL"
        else:
            return "FALLBACK"
    
    async def _execute_location_criteria(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute location-specific criteria using specialized executor"""
        logger.info("📍 Executing location criteria")
        
        execution_plan = policy_decision.get("execution_plan")
        result = await self.location_fee_executor.execute_location_criteria(execution_plan)
        
        # Enhance with policy information
        result["policy_context"] = {
            "decision_strategy": policy_decision.get("decision", {}).get("strategy"),
            "query_class": policy_decision.get("query_class"),
            "confidence": policy_decision.get("confidence", 0.8)
        }
        
        return result
    
    async def _execute_fee_criteria(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fee-specific criteria using specialized executor"""
        logger.info("💰 Executing fee criteria")
        
        execution_plan = policy_decision.get("execution_plan")
        result = await self.location_fee_executor.execute_fee_criteria(execution_plan)
        
        # Enhance with policy information
        result["policy_context"] = {
            "decision_strategy": policy_decision.get("decision", {}).get("strategy"),
            "query_class": policy_decision.get("query_class"),
            "confidence": policy_decision.get("confidence", 0.8)
        }
        
        return result
    
    async def _execute_combined_criteria(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute combined location and fee criteria using specialized executor"""
        logger.info("🎯 Executing combined location + fee criteria")
        
        execution_plan = policy_decision.get("execution_plan")
        result = await self.location_fee_executor.execute_combined_location_fee_criteria(execution_plan)
        
        # Enhance with policy information
        result["policy_context"] = {
            "decision_strategy": policy_decision.get("decision", {}).get("strategy"),
            "query_class": policy_decision.get("query_class"),
            "confidence": policy_decision.get("confidence", 0.8)
        }
        
        return result
    
    async def _execute_general_query(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute general queries using the main execution handler"""
        logger.info("🔍 Executing general query")
        
        result = await self.execution_handler.execute_plan(policy_decision)
        
        # Convert to execution system format
        return {
            "criteria_type": "GENERAL",
            "status": "SUCCESS" if result.get("execution_status") == "SUCCESS" else "ERROR",
            "results_count": result.get("results_count", 0),
            "results": result.get("results", []),
            "policy_context": {
                "decision_strategy": policy_decision.get("decision", {}).get("strategy"),
                "query_class": policy_decision.get("query_class"),
                "confidence": policy_decision.get("confidence", 0.8)
            },
            "execution_timestamp": datetime.now().isoformat()
        }
    
    async def _execute_fallback(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fallback query when no specific strategy applies"""
        logger.info("🔄 Executing fallback query")
        
        try:
            # Simple fallback - get some colleges from database
            if not self.db_manager.connection_status.get("mongodb"):
                await self.db_manager.initialize_connections()
            
            basic_results = await self.db_manager.execute_location_query(
                location_filter={},
                limit=10
            )
            
            # Format results
            formatted_results = []
            for college in basic_results:
                formatted_results.append({
                    "college_id": str(college.get("_id", "unknown")),
                    "name": college.get("Name", "Unknown College"),
                    "location": college.get("Location", "Unknown Location"),
                    "fees": college.get("Fees", "Contact college"),
                    "type": college.get("Type", "College"),
                    "programs": college.get("Departments", []),
                    "confidence": 0.6,
                    "source": "fallback_execution"
                })
            
            return {
                "criteria_type": "FALLBACK",
                "status": "SUCCESS",
                "results_count": len(formatted_results),
                "results": formatted_results,
                "policy_context": {
                    "decision_strategy": "FALLBACK",
                    "query_class": policy_decision.get("query_class", "FALLBACK"),
                    "confidence": 0.6
                },
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Fallback execution error: {e}")
            return self._build_error_response(policy_decision, f"Fallback execution failed: {e}")
    
    def _build_non_execution_response(self, policy_decision: Dict[str, Any]) -> Dict[str, Any]:
        """Build response for policy decisions that don't require execution"""
        decision = policy_decision.get("decision", {})
        
        return {
            "criteria_type": "NON_EXECUTION",
            "status": "COMPLETED",
            "action": decision.get("action", "UNKNOWN"),
            "reason": decision.get("reason", "No execution required"),
            "results_count": 0,
            "results": [],
            "policy_context": {
                "decision_strategy": decision.get("strategy"),
                "query_class": policy_decision.get("query_class"),
                "confidence": policy_decision.get("confidence", 0.8)
            },
            "execution_timestamp": datetime.now().isoformat()
        }
    
    def _build_error_response(self, policy_decision: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """Build error response for failed executions"""
        return {
            "criteria_type": "ERROR",
            "status": "EXECUTION_FAILED",
            "error_message": error_message,
            "results_count": 0,
            "results": [],
            "policy_context": {
                "policy_id": policy_decision.get("policy_id"),
                "query_class": policy_decision.get("query_class")
            },
            "execution_timestamp": datetime.now().isoformat()
        }
    
    async def get_execution_stats(self) -> Dict[str, Any]:
        """Get execution system statistics"""
        try:
            db_stats = await self.db_manager.get_database_stats()
            
            return {
                "system_status": "OPERATIONAL" if self.initialized else "NOT_INITIALIZED",
                "database_stats": db_stats,
                "supported_criteria": ["LOCATION", "FEE", "LOCATION_AND_FEE", "GENERAL"],
                "execution_strategies": ["LOCATION_ONLY", "FEE_ONLY", "LOCATION_AND_FEE", "GENERAL", "FALLBACK"],
                "system_info": {
                    "version": "1.0.0",
                    "specialized_executors": ["LocationFeeCriteriaExecutor"],
                    "database_connections": ["MongoDB", "FAISS"],
                    "last_check": datetime.now().isoformat()
                }
            }
            
        except Exception as e:
            return {
                "system_status": "ERROR",
                "error": str(e),
                "last_check": datetime.now().isoformat()
            }
    
    async def close(self):
        """Clean shutdown of execution system"""
        try:
            await self.db_manager.close_connections()
            self.initialized = False
            logger.info("🔒 Execution System shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

# Global execution system instance
execution_system = ExecutionSystem()