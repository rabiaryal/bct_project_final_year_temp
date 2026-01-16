"""
Location & Fee Criteria Executor
Specialized executor for location and fee-based college queries with optimized database operations
"""

from typing import Dict, Any, List, Optional, Tuple
import asyncio
import logging
from datetime import datetime
import re

from .database_manager import db_manager
from ..utils.logger import get_logger

logger = get_logger(__name__)

class LocationFeeCriteriaExecutor:
    """
    Specialized executor for location and fee criteria queries
    Optimizes database operations specifically for these two most common search criteria
    """
    
    def __init__(self):
        self.db_manager = db_manager
        self.initialized = False
    
    async def initialize(self):
        """Initialize the executor and database connections"""
        if not self.initialized:
            await self.db_manager.initialize_connections()
            self.initialized = True
            logger.info("✅ LocationFeeCriteriaExecutor initialized")
    
    async def execute_location_criteria(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute location-specific search criteria
        Handles various location query types: exact, partial, nearby
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info("📍 Executing location criteria search")
        
        try:
            # Extract location parameters from execution plan
            mongodb_config = execution_plan.get("mongodb", {})
            location_filter = mongodb_config.get("filter", {})
            
            # Execute location query with optimization
            results = await self.db_manager.execute_location_query(
                location_filter=location_filter,
                limit=20
            )
            
            # Post-process location results
            processed_results = await self._process_location_results(results, execution_plan)
            
            return {
                "criteria_type": "LOCATION",
                "status": "SUCCESS",
                "results_count": len(processed_results),
                "results": processed_results,
                "location_details": self._extract_location_summary(processed_results),
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Location criteria execution error: {e}")
            return self._build_error_response("LOCATION", str(e))
    
    async def execute_fee_criteria(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute fee-specific search criteria
        Handles fee ranges, maximum fees, budget-based searches
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info("💰 Executing fee criteria search")
        
        try:
            # Extract fee parameters from execution plan
            mongodb_config = execution_plan.get("mongodb", {})
            fee_filter = mongodb_config.get("filter", {})
            
            # Execute fee query with optimization
            results = await self.db_manager.execute_fee_query(
                fee_filter=fee_filter,
                limit=20
            )
            
            # Post-process fee results
            processed_results = await self._process_fee_results(results, execution_plan)
            
            return {
                "criteria_type": "FEE",
                "status": "SUCCESS",
                "results_count": len(processed_results),
                "results": processed_results,
                "fee_summary": self._extract_fee_summary(processed_results),
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Fee criteria execution error: {e}")
            return self._build_error_response("FEE", str(e))
    
    async def execute_combined_location_fee_criteria(self, execution_plan: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute combined location and fee criteria search
        This is the most optimized and commonly used search type
        """
        if not self.initialized:
            await self.initialize()
        
        logger.info("🎯 Executing combined location + fee criteria search")
        
        try:
            # Extract parameters from execution plan
            mongodb_config = execution_plan.get("mongodb", {})
            filter_conditions = mongodb_config.get("filter", {})
            
            # Separate location and fee filters
            location_filter, fee_filter = self._separate_location_fee_filters(filter_conditions)
            
            # Execute combined optimized query
            results = await self.db_manager.execute_combined_location_fee_query(
                location_filter=location_filter,
                fee_filter=fee_filter,
                limit=20
            )
            
            # Post-process combined results
            processed_results = await self._process_combined_results(results, execution_plan)
            
            return {
                "criteria_type": "LOCATION_AND_FEE",
                "status": "SUCCESS",
                "results_count": len(processed_results),
                "results": processed_results,
                "location_summary": self._extract_location_summary(processed_results),
                "fee_summary": self._extract_fee_summary(processed_results),
                "combined_insights": self._generate_combined_insights(processed_results),
                "execution_timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"❌ Combined criteria execution error: {e}")
            return self._build_error_response("LOCATION_AND_FEE", str(e))
    
    async def _process_location_results(self, raw_results: List[Dict], execution_plan: Dict) -> List[Dict]:
        """Process and format location-specific results"""
        processed_results = []
        
        for college in raw_results:
            # Standard formatting
            formatted_college = await self._format_college_result(college)
            
            # Add location-specific enhancements
            formatted_college.update({
                "location_relevance_score": self._calculate_location_relevance(college, execution_plan),
                "location_details": self._extract_location_details(college),
                "distance_category": self._categorize_location(college.get("Location", "")),
            })
            
            processed_results.append(formatted_college)
        
        # Sort by location relevance
        processed_results.sort(key=lambda x: x.get("location_relevance_score", 0.0), reverse=True)
        return processed_results[:15]  # Top 15 results
    
    async def _process_fee_results(self, raw_results: List[Dict], execution_plan: Dict) -> List[Dict]:
        """Process and format fee-specific results"""
        processed_results = []
        
        for college in raw_results:
            # Standard formatting
            formatted_college = await self._format_college_result(college)
            
            # Add fee-specific enhancements
            formatted_college.update({
                "fee_value": self._extract_fee_value(college),
                "fee_category": self._categorize_fee(college.get("Fees", "")),
                "fee_affordability_score": self._calculate_fee_affordability(college, execution_plan),
                "fee_details": self._extract_fee_details(college),
            })
            
            processed_results.append(formatted_college)
        
        # Sort by fee affordability (lower fees generally better)
        processed_results.sort(key=lambda x: (
            -x.get("fee_affordability_score", 0.0),  # Higher affordability score first
            x.get("fee_value", float('inf'))  # Lower fee value first
        ))
        return processed_results[:15]  # Top 15 results
    
    async def _process_combined_results(self, raw_results: List[Dict], execution_plan: Dict) -> List[Dict]:
        """Process and format combined location+fee results"""
        processed_results = []
        
        for college in raw_results:
            # Standard formatting
            formatted_college = await self._format_college_result(college)
            
            # Add both location and fee enhancements
            formatted_college.update({
                # Location enhancements
                "location_relevance_score": self._calculate_location_relevance(college, execution_plan),
                "location_details": self._extract_location_details(college),
                "distance_category": self._categorize_location(college.get("Location", "")),
                
                # Fee enhancements
                "fee_value": self._extract_fee_value(college),
                "fee_category": self._categorize_fee(college.get("Fees", "")),
                "fee_affordability_score": self._calculate_fee_affordability(college, execution_plan),
                "fee_details": self._extract_fee_details(college),
                
                # Combined score
                "overall_match_score": self._calculate_combined_match_score(college, execution_plan)
            })
            
            processed_results.append(formatted_college)
        
        # Sort by combined match score
        processed_results.sort(key=lambda x: x.get("overall_match_score", 0.0), reverse=True)
        return processed_results[:12]  # Top 12 results for combined search
    
    def _separate_location_fee_filters(self, filter_conditions: Dict[str, Any]) -> Tuple[Dict, Dict]:
        """Separate combined filters into location and fee specific filters"""
        location_filter = {}
        fee_filter = {}
        
        for field, condition in filter_conditions.items():
            if field.lower() in ["location"]:
                location_filter[field] = condition
            elif field.lower() in ["fees", "fee"]:
                fee_filter[field] = condition
            else:
                # Default: include in both (might be relevant for either)
                location_filter[field] = condition
                fee_filter[field] = condition
        
        return location_filter, fee_filter
    
    async def _format_college_result(self, college_data: Dict) -> Dict[str, Any]:
        """Format a single college result with standard fields"""
        return {
            "college_id": str(college_data.get("_id", "unknown")),
            "name": college_data.get("Name", "Unknown College"),
            "location": college_data.get("Location", "Unknown Location"),
            "fees": college_data.get("Fees", "Contact college"),
            "type": college_data.get("Type", "College"),
            "programs": college_data.get("Departments", []),
            "contact": college_data.get("ContactNumber", ""),
            "website": college_data.get("Website", ""),
            "confidence": 0.85,  # Default confidence for database results
            "source": "mongodb_execution",
            "timestamp": datetime.now().isoformat()
        }
    
    def _calculate_location_relevance(self, college: Dict, execution_plan: Dict) -> float:
        """Calculate location relevance score for a college"""
        score = 0.5  # Base score
        
        location = college.get("Location", "").lower()
        
        # Check for specific location preferences in execution plan
        mongodb_filter = execution_plan.get("mongodb", {}).get("filter", {})
        
        if "Location" in mongodb_filter:
            location_condition = mongodb_filter["Location"]
            
            if isinstance(location_condition, dict) and "$regex" in location_condition:
                search_location = location_condition["$regex"].lower()
                
                # Exact match boost
                if search_location in location:
                    score += 0.4
                
                # Popular location boost
                if "kathmandu" in search_location or "kathmandu" in location:
                    score += 0.1
                
                # Valley location boost
                valley_locations = ["kathmandu", "lalitpur", "bhaktapur"]
                if any(loc in location for loc in valley_locations):
                    score += 0.05
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_fee_affordability(self, college: Dict, execution_plan: Dict) -> float:
        """Calculate fee affordability score for a college"""
        score = 0.5  # Base score
        
        # Extract fee value
        fee_value = self._extract_fee_value(college)
        
        if fee_value == 0:
            return score  # No fee info, return base score
        
        # Check for fee preferences in execution plan
        mongodb_filter = execution_plan.get("mongodb", {}).get("filter", {})
        
        # Affordability categories
        if fee_value < 300000:  # Under 3 lakhs
            score += 0.4
        elif fee_value < 500000:  # Under 5 lakhs
            score += 0.3
        elif fee_value < 800000:  # Under 8 lakhs
            score += 0.2
        elif fee_value < 1200000:  # Under 12 lakhs
            score += 0.1
        
        return min(score, 1.0)  # Cap at 1.0
    
    def _calculate_combined_match_score(self, college: Dict, execution_plan: Dict) -> float:
        """Calculate overall match score for combined criteria"""
        location_score = self._calculate_location_relevance(college, execution_plan)
        fee_score = self._calculate_fee_affordability(college, execution_plan)
        
        # Weighted combination (fee slightly more important)
        combined_score = (location_score * 0.45) + (fee_score * 0.55)
        
        return combined_score
    
    def _extract_fee_value(self, college: Dict) -> float:
        """Extract numeric fee value from college data"""
        fees_str = college.get("Fees", college.get("fees", "0"))
        
        if not fees_str or fees_str == "Contact college":
            return 0.0
        
        # Extract numeric values from fee string
        numbers = re.findall(r'[\d,]+', str(fees_str))
        if numbers:
            try:
                # Remove commas and convert to float
                fee_value = float(numbers[0].replace(',', ''))
                return fee_value
            except ValueError:
                pass
        
        return 0.0
    
    def _categorize_location(self, location: str) -> str:
        """Categorize location for better organization"""
        location_lower = location.lower()
        
        if "kathmandu" in location_lower:
            return "Kathmandu Valley - Core"
        elif any(area in location_lower for area in ["lalitpur", "bhaktapur"]):
            return "Kathmandu Valley - Extended"
        elif any(city in location_lower for city in ["pokhara", "chitwan", "biratnagar"]):
            return "Major City"
        else:
            return "Regional"
    
    def _categorize_fee(self, fees: str) -> str:
        """Categorize fee for better organization"""
        fee_value = self._extract_fee_value({"Fees": fees})
        
        if fee_value == 0:
            return "Contact Required"
        elif fee_value < 300000:
            return "Budget Friendly (< 3L)"
        elif fee_value < 500000:
            return "Moderate (3L - 5L)"
        elif fee_value < 800000:
            return "Premium (5L - 8L)"
        else:
            return "High-End (8L+)"
    
    def _extract_location_details(self, college: Dict) -> Dict[str, Any]:
        """Extract detailed location information"""
        location = college.get("Location", "")
        
        return {
            "full_location": location,
            "category": self._categorize_location(location),
            "is_valley": any(area in location.lower() for area in ["kathmandu", "lalitpur", "bhaktapur"]),
            "is_major_city": any(city in location.lower() for city in ["pokhara", "chitwan", "biratnagar"])
        }
    
    def _extract_fee_details(self, college: Dict) -> Dict[str, Any]:
        """Extract detailed fee information"""
        fees_str = college.get("Fees", "")
        fee_value = self._extract_fee_value(college)
        
        return {
            "fee_text": fees_str,
            "fee_amount": fee_value,
            "category": self._categorize_fee(fees_str),
            "is_affordable": fee_value > 0 and fee_value < 500000,
            "formatted_fee": f"NPR {fee_value:,.0f}" if fee_value > 0 else "Contact required"
        }
    
    def _extract_location_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate location summary from results"""
        if not results:
            return {}
        
        location_categories = {}
        valley_count = 0
        total_count = len(results)
        
        for result in results:
            category = result.get("distance_category", "Unknown")
            location_categories[category] = location_categories.get(category, 0) + 1
            
            if result.get("location_details", {}).get("is_valley", False):
                valley_count += 1
        
        return {
            "total_colleges": total_count,
            "valley_colleges": valley_count,
            "location_distribution": location_categories,
            "valley_percentage": round((valley_count / total_count) * 100, 1) if total_count > 0 else 0
        }
    
    def _extract_fee_summary(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate fee summary from results"""
        if not results:
            return {}
        
        fee_values = []
        fee_categories = {}
        affordable_count = 0
        
        for result in results:
            fee_value = result.get("fee_value", 0)
            if fee_value > 0:
                fee_values.append(fee_value)
            
            category = result.get("fee_category", "Unknown")
            fee_categories[category] = fee_categories.get(category, 0) + 1
            
            if result.get("fee_details", {}).get("is_affordable", False):
                affordable_count += 1
        
        avg_fee = sum(fee_values) / len(fee_values) if fee_values else 0
        min_fee = min(fee_values) if fee_values else 0
        max_fee = max(fee_values) if fee_values else 0
        
        return {
            "total_colleges": len(results),
            "colleges_with_fee_info": len(fee_values),
            "average_fee": round(avg_fee, 2),
            "min_fee": min_fee,
            "max_fee": max_fee,
            "affordable_count": affordable_count,
            "fee_distribution": fee_categories,
            "affordable_percentage": round((affordable_count / len(results)) * 100, 1) if results else 0
        }
    
    def _generate_combined_insights(self, results: List[Dict]) -> Dict[str, Any]:
        """Generate insights for combined location+fee search"""
        if not results:
            return {}
        
        # Find best value colleges (good location + affordable fee)
        best_value = []
        for result in results:
            location_score = result.get("location_relevance_score", 0)
            fee_score = result.get("fee_affordability_score", 0)
            
            if location_score >= 0.7 and fee_score >= 0.7:
                best_value.append(result["name"])
        
        # Valley + affordable colleges
        valley_affordable = []
        for result in results:
            is_valley = result.get("location_details", {}).get("is_valley", False)
            is_affordable = result.get("fee_details", {}).get("is_affordable", False)
            
            if is_valley and is_affordable:
                valley_affordable.append(result["name"])
        
        return {
            "best_value_colleges": best_value[:5],  # Top 5 best value
            "valley_affordable_colleges": valley_affordable[:3],  # Top 3 valley + affordable
            "search_optimization": "Combined location and fee criteria applied",
            "recommendation": self._generate_recommendation(results)
        }
    
    def _generate_recommendation(self, results: List[Dict]) -> str:
        """Generate a recommendation based on search results"""
        if not results:
            return "No colleges found matching your criteria. Try expanding your search parameters."
        
        top_result = results[0]
        total_results = len(results)
        
        location_cat = top_result.get("distance_category", "Unknown")
        fee_cat = top_result.get("fee_category", "Unknown")
        
        recommendation = f"Found {total_results} colleges matching your criteria. "
        
        if top_result.get("overall_match_score", 0) > 0.8:
            recommendation += f"'{top_result['name']}' appears to be an excellent match "
        elif total_results > 8:
            recommendation += "You have many good options to choose from. "
        
        recommendation += f"Most results are in {location_cat} with {fee_cat} fees."
        
        return recommendation
    
    def _build_error_response(self, criteria_type: str, error_message: str) -> Dict[str, Any]:
        """Build error response for failed executions"""
        return {
            "criteria_type": criteria_type,
            "status": "ERROR",
            "error_message": error_message,
            "results_count": 0,
            "results": [],
            "execution_timestamp": datetime.now().isoformat()
        }

# Global executor instance
location_fee_executor = LocationFeeCriteriaExecutor()