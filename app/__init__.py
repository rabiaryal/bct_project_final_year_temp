"""
App Module: Application layer for the chatbot
- unified_pipeline: Central routing controller with hierarchical intent detection
"""

from .unified_pipeline import UnifiedPipelineController, create_unified_pipeline

# Backward compatibility aliases
PipelineController = UnifiedPipelineController
create_pipeline_controller = create_unified_pipeline

__all__ = [
    'UnifiedPipelineController', 
    'create_unified_pipeline',
    'PipelineController',  # Backward compatibility
    'create_pipeline_controller'  # Backward compatibility
]
