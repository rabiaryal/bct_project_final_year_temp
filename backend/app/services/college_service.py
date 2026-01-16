"""College Service Layer"""

from typing import Dict, List, Any, Optional
from app.repositories.mongo_client import MongoRepository
from app.utils.logger import get_logger

logger = get_logger(__name__)

class CollegeService:
    """Business logic for college operations"""
    
    def __init__(self, mongo_repo: MongoRepository):
        self.mongo_repo = mongo_repo
    
    async def search_colleges(self, search_params: Dict[str, Any]) -> Dict[str, Any]:
        """Search for colleges with enhanced logic"""
        try:
            # Clean and validate search parameters
            cleaned_params = self._clean_search_params(search_params)
            
            # Perform search
            colleges = await self.mongo_repo.search_colleges(cleaned_params)
            
            # Enhance results with additional info
            enhanced_results = []
            for college in colleges:
                enhanced_college = await self._enhance_college_info(college)
                enhanced_results.append(enhanced_college)
            
            return {
                "success": True,
                "results": enhanced_results,
                "count": len(enhanced_results),
                "search_params": cleaned_params
            }
            
        except Exception as e:
            logger.error(f"College search service error: {e}")
            return {
                "success": False,
                "error": str(e),
                "results": [],
                "count": 0
            }
    
    async def search_by_name(self, college_name: str) -> Dict[str, Any]:
        """Search for a college by name"""
        try:
            if not self.mongo_repo:
                return None
                
            # Search for college by name (case-insensitive)
            query = {"Name": {"$regex": college_name, "$options": "i"}}
            colleges = await self.mongo_repo.find_colleges(query, limit=1)
            
            if colleges:
                return colleges[0]  # Return first match
            return None
            
        except Exception as e:
            logger.error(f"Error searching college by name: {e}")
            return None
    
    async def get_college_details(self, college_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific college"""
        try:
            college = await self.mongo_repo.find_college_by_name(college_name)
            
            if not college:
                return {
                    "success": False,
                    "message": f"College '{college_name}' not found",
                    "suggestions": await self._get_similar_colleges(college_name)
                }
            
            # Enhance college information
            enhanced_college = await self._enhance_college_info(college)
            
            return {
                "success": True,
                "college": enhanced_college
            }
            
        except Exception as e:
            logger.error(f"College details service error: {e}")
            return {
                "success": False,
                "error": str(e)
            }
    
    async def get_college_admission_info(self, college_name: str) -> Dict[str, Any]:
        """Get admission-specific information"""
        try:
            college = await self.mongo_repo.find_college_by_name(college_name)
            
            if not college:
                return {"success": False, "message": "College not found"}
            
            admission_info = {
                "college_name": college.get("name"),
                "admission_process": college.get("admission_process", "Contact college for details"),
                "eligibility": college.get("eligibility", "Please check college requirements"),
                "entrance_exam": college.get("entrance_exam", "Information not available"),
                "application_deadline": college.get("application_deadline", "Contact college"),
                "contact_info": {
                    "phone": college.get("phone"),
                    "email": college.get("email"),
                    "website": college.get("website")
                }
            }
            
            return {"success": True, "admission_info": admission_info}
            
        except Exception as e:
            logger.error(f"Admission info service error: {e}")
            return {"success": False, "error": str(e)}
    
    async def get_college_fees(self, college_name: str, course: str = None) -> Dict[str, Any]:
        """Get fee information for college/course"""
        try:
            college = await self.mongo_repo.find_college_by_name(college_name)
            
            if not college:
                return {"success": False, "message": "College not found"}
            
            fee_info = {
                "college_name": college.get("name"),
                "general_fees": college.get("fees", "Contact college for fee details"),
                "course_specific": course,
                "scholarships": college.get("scholarships", "Information not available"),
                "payment_options": college.get("payment_options", "Contact college")
            }
            
            return {"success": True, "fee_info": fee_info}
            
        except Exception as e:
            logger.error(f"Fee info service error: {e}")
            return {"success": False, "error": str(e)}
    
    def _clean_search_params(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize search parameters"""
        cleaned = {}
        
        for key, value in params.items():
            if value and isinstance(value, str):
                cleaned[key] = value.strip()
            elif value:
                cleaned[key] = value
        
        return cleaned
    
    async def _enhance_college_info(self, college: Dict[str, Any]) -> Dict[str, Any]:
        """Enhance college information with additional fields"""
        enhanced = college.copy()
        
        # Add computed fields
        enhanced["has_website"] = bool(college.get("website"))
        enhanced["has_email"] = bool(college.get("email"))
        enhanced["has_phone"] = bool(college.get("phone"))
        
        # Extract key information
        enhanced["key_info"] = {
            "type": college.get("type", "Unknown"),
            "affiliation": college.get("affiliation", "Information not available"),
            "established": college.get("established", "Information not available")
        }
        
        return enhanced
    
    async def _get_similar_colleges(self, college_name: str) -> List[str]:
        """Get suggestions for similar college names"""
        try:
            # Simple fuzzy search for suggestions
            query = {"name": {"$regex": college_name.split()[0], "$options": "i"}}
            cursor = self.mongo_repo.collection.find(query, {"name": 1}).limit(3)
            results = await cursor.to_list(length=3)
            
            return [college["name"] for college in results]
            
        except Exception:
            return []