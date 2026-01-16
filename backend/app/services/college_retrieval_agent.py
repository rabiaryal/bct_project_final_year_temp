"""
College Information Retrieval Agent with Entity Role Integration
Combines FAISS semantic search with MongoDB metadata filtering using structured entities
"""

import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import asyncio
import sys
import os

# Add backend to path  
sys.path.append('/Applications/development/ml learning/bct_final_year_project/backend')

from app.repositories.mongo_client import MongoRepository
from app.utils.logger import get_logger
from app.policy.entity_roles import entity_processor, EntityRole, Entity

logger = get_logger(__name__)

class CollegeRetrievalAgent:
    """
    Intelligent College Information Retrieval Agent
    Combines FAISS semantic search with MongoDB metadata filtering
    """
    
    def __init__(self, faiss_index_path: str = None, mongo_repo: MongoRepository = None):
        self.faiss_index_path = faiss_index_path or "/Applications/development/ml learning/bct_final_year_project/models/faiss_index"
        self.mongo_repo = mongo_repo
        self.faiss_index = None
        self.embeddings_metadata = None
        self.context_store = {}
        self.confidence_threshold = 0.05  # Very low threshold for semantic search
        
    async def initialize(self):
        """Initialize FAISS index and MongoDB connection"""
        try:
            # Initialize FAISS index
            await self._load_faiss_index()
            
            # Initialize MongoDB if not provided
            if self.mongo_repo is None:
                self.mongo_repo = MongoRepository()
                await self.mongo_repo.connect()
                
            logger.info("College Retrieval Agent initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize retrieval agent: {e}")
            return False
    
    async def _load_faiss_index(self):
        """Load FAISS index and metadata"""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # Load FAISS index
            index_file = os.path.join(self.faiss_index_path, "faiss.index")
            if os.path.exists(index_file):
                self.faiss_index = faiss.read_index(index_file)
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            else:
                logger.warning(f"FAISS index not found at {index_file}")
                # Create a simple demo index for testing
                await self._create_demo_index()
            
            # Load embeddings metadata
            metadata_file = os.path.join(self.faiss_index_path, "metadata.json")
            if os.path.exists(metadata_file):
                with open(metadata_file, 'r') as f:
                    self.embeddings_metadata = json.load(f)
                logger.info(f"Loaded metadata for {len(self.embeddings_metadata)} colleges")
            else:
                logger.warning("No embeddings metadata found, generating from MongoDB...")
                self.embeddings_metadata = await self._generate_metadata_from_mongodb()
                if self.embeddings_metadata:
                    # Save generated metadata for next time
                    await self._save_metadata(metadata_file)
                    logger.info(f"Generated and saved metadata for {len(self.embeddings_metadata)} colleges")
                else:
                    logger.warning("Could not generate metadata, using demo fallback")
                    self.embeddings_metadata = []
            
            # Initialize embedding model
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Validate FAISS index matches metadata size
            if hasattr(self, 'faiss_index') and self.faiss_index is not None:
                expected_vectors = len(self.embeddings_metadata)
                actual_vectors = self.faiss_index.ntotal
                
                if actual_vectors != expected_vectors:
                    logger.warning(f"FAISS index size mismatch: {actual_vectors} vectors vs {expected_vectors} metadata entries")
                    logger.info("Rebuilding FAISS index to match current metadata...")
                    await self._rebuild_faiss_index_from_metadata()
                else:
                    logger.info(f"FAISS index matches metadata: {actual_vectors} vectors")
            
        except ImportError as e:
            logger.error(f"Missing dependencies: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading FAISS index: {e}")
            raise
    
    async def _create_demo_index(self):
        """Create a demo FAISS index for testing"""
        try:
            import faiss
            from sentence_transformers import SentenceTransformer
            
            # Sample college descriptions for demo
            demo_colleges = [
                {
                    "id": "demo_1", 
                    "name": "Kathmandu University",
                    "description": "Premier university in Nepal offering engineering, management and computer science programs",
                    "location": "Kathmandu",
                    "programs": ["BE Computer", "MBA", "BBA"],
                    "department": "Engineering"
                },
                {
                    "id": "demo_2",
                    "name": "Tribhuvan University", 
                    "description": "Largest university in Nepal with multiple campuses offering diverse academic programs",
                    "location": "Kathmandu",
                    "programs": ["BSc CSIT", "BE Civil", "BE Computer"],
                    "department": "Engineering"
                },
                {
                    "id": "demo_3",
                    "name": "Pokhara University",
                    "description": "University in Pokhara offering engineering and management programs with excellent facilities",
                    "location": "Pokhara", 
                    "programs": ["BE Computer", "BE Civil", "BBA"],
                    "department": "Engineering"
                }
            ]
            
            # Generate embeddings
            embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            descriptions = [college["description"] for college in demo_colleges]
            embeddings = embedding_model.encode(descriptions)
            
            # Create FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for similarity
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Store metadata
            self.embeddings_metadata = demo_colleges
            
            # Save to disk
            os.makedirs(self.faiss_index_path, exist_ok=True)
            faiss.write_index(self.faiss_index, os.path.join(self.faiss_index_path, "faiss.index"))
            
            with open(os.path.join(self.faiss_index_path, "metadata.json"), 'w') as f:
                json.dump(demo_colleges, f, indent=2)
            
            logger.info("Created demo FAISS index successfully")
            
        except Exception as e:
            logger.error(f"Failed to create demo index: {e}")
            raise
    
    async def _generate_metadata_from_mongodb(self) -> List[Dict]:
        """Generate metadata from MongoDB college data"""
        try:
            # We need MongoDB connection first
            if self.mongo_repo is None:
                self.mongo_repo = MongoRepository()
                await self.mongo_repo.connect()
            
            # Get all college data from MongoDB
            colleges = await self.mongo_repo.find_colleges({}, limit=1000)
            
            if not colleges:
                logger.warning("No college data found in MongoDB")
                return []
            
            metadata = []
            for i, college in enumerate(colleges):
                # Create metadata entry that matches FAISS index order
                metadata_entry = {
                    "id": str(college.get("_id", f"college_{i}")),
                    "name": college.get("Name", "Unknown"),  # MongoDB uses "Name" with capital N
                    "description": f"{college.get('Name', 'Unknown')} - {college.get('Type', 'College')} located in {college.get('Location', 'Unknown')}",
                    "location": college.get("Location", "Unknown"),  # MongoDB uses "Location" with capital L  
                    "type": college.get("Type", ""),  # MongoDB uses "Type" with capital T
                    "affiliation": college.get("affiliation", ""),
                    "established": college.get("established", ""),
                    "courses": college.get("courses", ""),
                    "website": college.get("website", ""),
                    "phone": college.get("ContactNumber", ""),  # MongoDB uses "ContactNumber"
                    "vector_index": i  # Maps to FAISS vector position
                }
                metadata.append(metadata_entry)
            
            logger.info(f"Generated metadata for {len(metadata)} colleges")
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata from MongoDB: {e}")
            return []
    
    async def _rebuild_faiss_index_from_metadata(self):
        """Rebuild FAISS index from current metadata"""
        try:
            import faiss
            
            if not self.embeddings_metadata:
                logger.error("Cannot rebuild FAISS index: no metadata available")
                return
                
            # Generate embeddings for each college description
            descriptions = []
            for college in self.embeddings_metadata:
                desc = college.get('description', '')
                if not desc:
                    # Create description from available fields
                    name = college.get('name', 'Unknown')
                    location = college.get('location', 'Unknown')
                    college_type = college.get('type', 'College')
                    desc = f"{name} - {college_type} located in {location}"
                descriptions.append(desc)
            
            logger.info(f"Generating embeddings for {len(descriptions)} college descriptions...")
            embeddings = self.embedding_model.encode(descriptions)
            
            # Create new FAISS index
            dimension = embeddings.shape[1]
            self.faiss_index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
            self.faiss_index.add(embeddings.astype('float32'))
            
            # Save to disk
            os.makedirs(self.faiss_index_path, exist_ok=True)
            faiss.write_index(self.faiss_index, os.path.join(self.faiss_index_path, "faiss.index"))
            
            logger.info(f"Successfully rebuilt FAISS index with {self.faiss_index.ntotal} vectors")
            
        except Exception as e:
            logger.error(f"Failed to rebuild FAISS index: {e}")
            raise
    
    async def _save_metadata(self, metadata_file: str):
        """Save metadata to file"""
        try:
            os.makedirs(os.path.dirname(metadata_file), exist_ok=True)
            with open(metadata_file, 'w', encoding='utf-8') as f:
                json.dump(self.embeddings_metadata, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved metadata to {metadata_file}")
        except Exception as e:
            logger.error(f"Error saving metadata: {e}")
    
    async def process_query(self, user_query: str, entities_from_ner: List[Dict]) -> Dict:
        """
        Main processing function following the specified workflow
        
        Args:
            user_query: Raw user input
            entities_from_ner: List of extracted entities with type and confidence
            
        Returns:
            JSON response following the specified format
        """
        try:
            # Step 1: Update Context
            session_id = "default_session"  # In real system, this would be user-specific
            context = self._update_context(session_id, user_query, entities_from_ner)
            
            # Step 2: Policy Decision
            policy_decision = await self._make_policy_decision(user_query, entities_from_ner, context)
            
            # Step 3: Execute DB Action - Always use semantic search (FAISS)
            if policy_decision["action"] == "semantic_search":
                retrieved_results = await self._execute_semantic_search(
                    user_query, 
                    entities_from_ner,
                    policy_decision.get("filters", {}),
                    policy_decision.get("top_n", 5)
                )
            else:
                # For clarify_query or other actions
                retrieved_results = []
            
            # Step 4: Update Context with Results
            self._store_results_in_context(session_id, retrieved_results)
            
            # Step 5: Determine Next Action
            next_action = self._determine_next_action(retrieved_results, entities_from_ner)
            
            # Generate final response
            response = {
                "context": context,
                "policy": policy_decision["action"], 
                "db_action": {
                    "type": policy_decision["action"],
                    "query_embedding": await self._get_query_embedding(user_query),
                    "filters": policy_decision.get("filters", {}),
                    "top_n": policy_decision.get("top_n", 5)
                },
                "retrieved_results": retrieved_results,
                "next_action": next_action
            }
            
            return response
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "context": f"Error processing query: {str(e)}",
                "policy": "error",
                "db_action": {"type": "error", "query_embedding": None, "filters": {}, "top_n": 0},
                "retrieved_results": [],
                "next_action": "display_error"
            }
    
    def _update_context(self, session_id: str, user_query: str, entities: List[Dict]) -> str:
        """Update conversation context"""
        if session_id not in self.context_store:
            self.context_store[session_id] = {
                "queries": [],
                "entities": [],
                "last_results": [],
                "conversation_state": "initial"
            }
        
        # Add current query and entities
        self.context_store[session_id]["queries"].append({
            "query": user_query,
            "timestamp": datetime.now().isoformat(),
            "entities": entities
        })
        
        # Keep only last 5 queries for context
        self.context_store[session_id]["queries"] = self.context_store[session_id]["queries"][-5:]
        
        # Update conversation state
        entity_types = [e.get("type", "").lower() for e in entities]
        if "college_name" in entity_types:
            state = "specific_college_inquiry"
        elif "program" in entity_types or "department" in entity_types:
            state = "program_search"
        elif "location" in entity_types:
            state = "location_based_search"
        else:
            state = "general_inquiry"
        
        self.context_store[session_id]["conversation_state"] = state
        
        return f"User seeking {state} with entities: {[e.get('text', '') for e in entities]}"
    
    async def _make_policy_decision(self, user_query: str, entities: List[Dict], context: str) -> Dict:
        """Enhanced policy decision with entity role analysis"""
        # Convert to structured entities
        structured_entities = entity_processor.from_nlu_output(entities)
        
        # Get entity context for intelligent decision making
        entity_context = entity_processor.get_entity_context(structured_entities)
        
        # Build database query configuration
        db_query_config = entity_processor.build_database_query(structured_entities)
        
        # Enhanced decision logic based on entity roles
        if entity_context["has_identifier"]:
            action = "semantic_search"
            top_n = 3  # Specific lookup, fewer results
            reasoning = f"Identifier-driven query: {entity_context['query_strategy']}"
            
        elif entity_context["has_filters"] or entity_context["has_constraints"]:
            action = "semantic_search" 
            top_n = 10  # Filtered search, more results
            reasoning = f"Filter-driven search: {len(structured_entities)} entities"
            
        elif entity_context["has_signals"]:
            action = "semantic_search"
            top_n = 5  # Recommendation, moderate results
            reasoning = f"Signal-driven recommendation: {entity_context['query_strategy']}"
            
        elif len(structured_entities) > 0:
            action = "semantic_search"
            top_n = 5  # General entity-based search
            reasoning = f"Entity-based search: {entity_context['entity_count']} entities"
            
        else:
            action = "clarify_query"
            top_n = 0
            reasoning = "No entities detected, need clarification"
        
        return {
            "action": action,
            "filters": db_query_config["filter"],
            "top_n": top_n,
            "reasoning": reasoning,
            "entity_context": entity_context,
            "db_query_config": db_query_config,
            "structured_entities": structured_entities
        }
    
    async def _execute_semantic_search(self, user_query: str, entities: List[Dict], filters: Dict, top_n: int) -> List[Dict]:
        """Enhanced FAISS semantic search with entity role-based filtering"""
        try:
            # Convert to structured entities for better processing
            structured_entities = entity_processor.from_nlu_output(entities)
            entity_context = entity_processor.get_entity_context(structured_entities)
            
            # Generate query embedding
            query_vector = self.embedding_model.encode([user_query]).astype('float32')
            
            # Search FAISS index - adjust search space based on entity context
            search_multiplier = 4 if entity_context["has_filters"] or entity_context["has_constraints"] else 2
            search_k = min(top_n * search_multiplier, self.faiss_index.ntotal)
            scores, indices = self.faiss_index.search(query_vector, search_k)
            
            logger.info(f"FAISS search for '{user_query}' returned {len(indices[0])} candidates")
            logger.info(f"Entity strategy: {entity_context.get('query_strategy', 'general')}")
            
            results = []
            for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
                if idx >= len(self.embeddings_metadata):
                    logger.debug(f"Index {idx} out of range (max: {len(self.embeddings_metadata)})")
                    continue
                    
                college_meta = self.embeddings_metadata[idx]
                college_name = college_meta.get("name", "Unknown")
                
                # Enhanced confidence threshold based on entity context
                base_threshold = self.confidence_threshold
                
                # Dynamic threshold adjustment based on entity roles
                if entity_context["has_identifier"]:
                    # Very permissive for identifier matches
                    adjusted_threshold = 0.0
                    # Boost score for exact name matches
                    for entity in structured_entities:
                        if entity.role == EntityRole.IDENTIFIER and entity.value.lower() in college_name.lower():
                            score += 0.3  # Significant boost for name matches
                            break
                            
                elif entity_context["has_filters"]:
                    # Moderate threshold for filter-based searches
                    adjusted_threshold = base_threshold * 0.5
                    
                elif entity_context["has_constraints"]:
                    # Flexible threshold for constraint-based searches
                    adjusted_threshold = base_threshold * 0.7
                    
                else:
                    # Standard threshold for general queries
                    adjusted_threshold = base_threshold * 0.8
                
                logger.debug(f"Candidate {i+1}: {college_name} (score: {score:.3f}, threshold: {adjusted_threshold:.3f})")
                
                # Check confidence threshold
                if score < adjusted_threshold:
                    logger.debug(f"✗ Below threshold: {college_name} ({score:.3f} < {adjusted_threshold:.3f})")
                    continue
                
                # Add additional database info if available
                enhanced_info = await self._enhance_with_database_info(college_meta)
                
                # Apply entity-based filters using structured entities
                if structured_entities and not self._passes_entity_filters(enhanced_info, college_meta, structured_entities):
                    logger.debug(f"✗ Failed entity filters: {enhanced_info.get('Name', college_name)}")
                    continue
                
                result = {
                    "college_id": college_meta.get("id", f"college_{idx}"),
                    "name": enhanced_info.get("Name", enhanced_info.get("name", college_meta.get("name", "Unknown"))),
                    "location": enhanced_info.get("Location", enhanced_info.get("location", college_meta.get("location", "Unknown"))),
                    "programs": enhanced_info.get("programs", enhanced_info.get("Departments", college_meta.get("programs", []))),
                    "fees": enhanced_info.get("fees", "Contact college"),
                    "ranking": enhanced_info.get("ranking", "Not ranked"),
                    "confidence": float(score),
                    "description": college_meta.get("description", ""),
                    "source": "faiss_semantic_search",
                    "entity_matches": self._get_entity_matches(enhanced_info, structured_entities)
                }
                
                results.append(result)
                logger.info(f"✓ Added: {result['name']} (score: {score:.3f})")
                
                if len(results) >= top_n:
                    break
            
            logger.info(f"Semantic search completed: {len(results)} colleges found")
            return results
            
        except Exception as e:
            logger.error(f"Semantic search error: {e}")
            return []
    
    async def _execute_metadata_filter(self, entities: List[Dict], filters: Dict, top_n: int) -> List[Dict]:
        """Execute pure metadata-based filtering"""
        try:
            # Build MongoDB query from filters
            mongo_query = {}
            
            if filters.get("college_name"):
                mongo_query["Name"] = {"$regex": filters["college_name"], "$options": "i"}  # Add college name search
            if filters.get("location"):
                mongo_query["Location"] = {"$regex": filters["location"], "$options": "i"}  # MongoDB uses "Location"
            if filters.get("program"):
                mongo_query["Departments"] = {"$regex": filters["program"], "$options": "i"}  # MongoDB uses "Departments"
            if filters.get("department"):
                mongo_query["Departments"] = {"$regex": filters["department"], "$options": "i"}  # MongoDB uses "Departments"
            if filters.get("fees_max"):
                mongo_query["fees"] = {"$lte": filters["fees_max"]}
            
            # Query MongoDB
            if self.mongo_repo:
                colleges = await self.mongo_repo.find_colleges(mongo_query, limit=top_n)
            else:
                # Fallback to metadata filtering
                colleges = []
                for college_meta in self.embeddings_metadata:
                    if self._passes_filters(college_meta, filters):
                        colleges.append(college_meta)
                colleges = colleges[:top_n]
            
            # Format results
            results = []
            for college in colleges:
                result = {
                    "college_id": college.get("_id", college.get("id", "unknown")),
                    "name": college.get("Name", college.get("name", "Unknown")),  # MongoDB uses "Name" with capital N
                    "location": college.get("Location", college.get("location", "Unknown")),  # MongoDB uses "Location" with capital L
                    "programs": college.get("programs", college.get("Departments", [])),
                    "fees": college.get("fees", "Contact college"),
                    "ranking": college.get("ranking", "Not ranked"),
                    "confidence": 1.0,  # Exact match
                    "source": "metadata_filter"
                }
                results.append(result)
            
            return results
            
        except Exception as e:
            logger.error(f"Metadata filter error: {e}")
            return []
    
    def _passes_semantic_filters(self, enhanced_info: Dict, college_meta: Dict, filters: Dict) -> bool:
        """Check if FAISS result passes semantic filters"""
        # College name filter - use fuzzy matching for better recall
        if filters.get("college_name"):
            search_name = filters["college_name"].lower()
            college_name = enhanced_info.get("Name", college_meta.get("name", "")).lower()
            
            # Multiple matching strategies for better recall
            name_parts = search_name.split()
            college_parts = college_name.split()
            
            # Check if any significant word from search appears in college name
            matches = 0
            for part in name_parts:
                if len(part) > 3:  # Skip very short words
                    for college_part in college_parts:
                        if part in college_part or college_part in part:
                            matches += 1
                            break
            
            # Require at least half of significant words to match
            significant_words = [p for p in name_parts if len(p) > 3]
            if significant_words and matches / len(significant_words) < 0.5:
                return False
        
        # Location filter
        if filters.get("location"):
            location = enhanced_info.get("Location", college_meta.get("location", "")).lower()
            if filters["location"].lower() not in location:
                return False
        
        # Program filter - check both Departments and programs fields
        if filters.get("program"):
            departments = enhanced_info.get("Departments", college_meta.get("programs", []))
            if isinstance(departments, str):
                departments = [departments]
            elif isinstance(departments, list):
                departments = [str(d) for d in departments]
            else:
                departments = []
            
            program_text = " ".join(departments).lower()
            if filters["program"].lower() not in program_text:
                return False
        
        return True
    
    def _passes_filters(self, college_meta: Dict, filters: Dict) -> bool:
        """Check if college passes metadata filters"""
        if filters.get("location"):
            college_location = college_meta.get("location", "").lower()
            if filters["location"].lower() not in college_location:
                return False
        
        if filters.get("program"):
            college_programs = [p.lower() for p in college_meta.get("programs", [])]
            if not any(filters["program"].lower() in prog for prog in college_programs):
                return False
        
        if filters.get("department"):
            college_dept = college_meta.get("department", "").lower()
            if filters["department"].lower() not in college_dept:
                return False
        
        return True
    
    async def _enhance_with_database_info(self, college_meta: Dict) -> Dict:
        """Enhance college info with database data"""
        try:
            if self.mongo_repo:
                # Try to find matching college in database
                db_college = await self.mongo_repo.find_college_by_name(college_meta.get("name", ""))
                if db_college:
                    return {**college_meta, **db_college}
            
            return college_meta
            
        except Exception as e:
            logger.error(f"Database enhancement error: {e}")
            return college_meta
    
    async def _get_query_embedding(self, user_query: str) -> List[float]:
        """Generate embedding for user query"""
        try:
            embedding = self.embedding_model.encode([user_query])
            return embedding[0].tolist()
        except Exception as e:
            logger.error(f"Embedding generation error: {e}")
            return []
    
    def _store_results_in_context(self, session_id: str, results: List[Dict]):
        """Store search results in context for follow-up queries"""
        if session_id in self.context_store:
            self.context_store[session_id]["last_results"] = results
    
    def _determine_next_action(self, results: List[Dict], entities: List[Dict]) -> str:
        """Determine the next action based on results"""
        if not results:
            if not entities:
                return "ask_clarification"
            else:
                return "refine_query"
        elif len(results) == 1:
            return "display_single_result"
        elif len(results) <= 5:
            return "display_results"
        else:
            return "refine_query"
    
    def _passes_entity_filters(self, enhanced_info: Dict, college_meta: Dict, entities: List[Entity]) -> bool:
        """Enhanced entity-based filtering using structured entities"""
        for entity in entities:
            if not entity.is_valid():
                continue
                
            # Apply different filtering logic based on entity role
            if entity.role == EntityRole.IDENTIFIER:
                # Exact matching for identifiers (college names)
                college_name = enhanced_info.get("Name", college_meta.get("name", "")).lower()
                search_name = entity.value.lower()
                
                # Multiple matching strategies for identifiers
                if not (search_name in college_name or 
                       college_name in search_name or
                       self._fuzzy_match(search_name, college_name, threshold=0.8)):
                    logger.debug(f"Identifier mismatch: '{search_name}' not found in '{college_name}'")
                    return False
                    
            elif entity.role == EntityRole.FILTER:
                # Flexible matching for filters
                if not self._check_filter_match(enhanced_info, college_meta, entity):
                    logger.debug(f"Filter mismatch: {entity.type}={entity.value}")
                    return False
                    
            elif entity.role == EntityRole.CONSTRAINT:
                # Numeric/logical constraints
                if not self._check_constraint_match(enhanced_info, college_meta, entity):
                    logger.debug(f"Constraint mismatch: {entity.type}={entity.value}")
                    return False
        
        return True
    
    def _check_filter_match(self, enhanced_info: Dict, college_meta: Dict, entity: Entity) -> bool:
        """Check if entity filter matches college data"""
        if entity.type == "LOCATION":
            location = enhanced_info.get("Location", college_meta.get("location", "")).lower()
            search_location = entity.value.lower()
            return search_location in location or location in search_location
            
        elif entity.type in ["PROGRAM", "DEPARTMENT"]:
            departments = enhanced_info.get("Departments", college_meta.get("programs", []))
            if isinstance(departments, str):
                departments = [departments]
            elif isinstance(departments, list):
                departments = [str(d) for d in departments]
            else:
                departments = []
            
            program_text = " ".join(departments).lower()
            search_program = entity.value.lower()
            return search_program in program_text
            
        elif entity.type == "COLLEGE_TYPE":
            college_type = enhanced_info.get("Type", enhanced_info.get("college_type", "")).lower()
            search_type = entity.value.lower()
            return search_type in college_type
            
        elif entity.type == "FACILITY":
            facilities = enhanced_info.get("Facilities", enhanced_info.get("facilities", [])) 
            if isinstance(facilities, str):
                facilities = [facilities]
            facility_text = " ".join(str(f) for f in facilities).lower()
            search_facility = entity.value.lower()
            return search_facility in facility_text
        
        return True  # Default to pass if no specific logic
    
    def _check_constraint_match(self, enhanced_info: Dict, college_meta: Dict, entity: Entity) -> bool:
        """Check if entity constraint matches college data"""
        if entity.type in ["FEE", "FEES"]:
            # Fee constraint checking
            college_fees = enhanced_info.get("Fees", enhanced_info.get("fees"))
            if not college_fees:
                return True  # Pass if no fee information available
                
            try:
                # Extract numeric value from fee string
                import re
                fee_numbers = re.findall(r'\d+', str(college_fees))
                if fee_numbers:
                    college_fee_amount = int(fee_numbers[0])
                    
                    # Parse constraint value
                    constraint_value = entity.value.lower()
                    if 'below' in constraint_value or 'under' in constraint_value or '<' in constraint_value:
                        constraint_numbers = re.findall(r'\d+', constraint_value)
                        if constraint_numbers:
                            max_fee = int(constraint_numbers[0])
                            if 'lakh' in constraint_value:
                                max_fee *= 100000
                            return college_fee_amount <= max_fee
                    
            except Exception as e:
                logger.debug(f"Fee constraint parsing error: {e}")
                return True  # Pass on parsing errors
                
        elif entity.type == "SEATS":
            # Seat constraint checking
            seats = enhanced_info.get("Seats", enhanced_info.get("seats"))
            if seats:
                try:
                    college_seats = int(seats)
                    constraint_value = entity.value.lower()
                    if 'above' in constraint_value or '>' in constraint_value:
                        import re
                        numbers = re.findall(r'\d+', constraint_value)
                        if numbers:
                            min_seats = int(numbers[0])
                            return college_seats >= min_seats
                except Exception as e:
                    logger.debug(f"Seats constraint parsing error: {e}")
        
        return True  # Default to pass
    
    def _fuzzy_match(self, str1: str, str2: str, threshold: float = 0.8) -> bool:
        """Simple fuzzy string matching"""
        try:
            # Simple character-based similarity
            set1, set2 = set(str1.lower()), set(str2.lower())
            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            similarity = intersection / union if union > 0 else 0
            return similarity >= threshold
        except:
            return False
    
    def _get_entity_matches(self, enhanced_info: Dict, entities: List[Entity]) -> Dict[str, Any]:
        """Get information about which entities matched"""
        matches = {
            "identifiers": [],
            "filters": [],
            "constraints": [],
            "total_matches": 0
        }
        
        for entity in entities:
            if entity.role == EntityRole.IDENTIFIER:
                college_name = enhanced_info.get("Name", "").lower()
                if entity.value.lower() in college_name:
                    matches["identifiers"].append(entity.value)
                    matches["total_matches"] += 1
                    
            elif entity.role == EntityRole.FILTER:
                if self._check_filter_match(enhanced_info, {}, entity):
                    matches["filters"].append(f"{entity.type}:{entity.value}")
                    matches["total_matches"] += 1
                    
            elif entity.role == EntityRole.CONSTRAINT:
                if self._check_constraint_match(enhanced_info, {}, entity):
                    matches["constraints"].append(f"{entity.type}:{entity.value}")
                    matches["total_matches"] += 1
        
        return matches