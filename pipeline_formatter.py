#!/usr/bin/env python3
"""
Enhanced Terminal Output Formatter for Pipeline Stages
"""

import re
from datetime import datetime
from typing import Dict, Any, List

class PipelineLogFormatter:
    """Format pipeline logs for better readability"""
    
    def __init__(self):
        self.stage_colors = {
            "NLU": "\033[94m",  # Blue
            "RETRIEVAL": "\033[92m",  # Green
            "CONTEXT": "\033[93m",  # Yellow
            "POLICY": "\033[95m",  # Magenta
            "ACTION": "\033[96m",  # Cyan
            "RESPONSE": "\033[91m",  # Red
            "COMPLETE": "\033[92m",  # Green
            "ERROR": "\033[91m",  # Red
            "RESET": "\033[0m"  # Reset
        }
    
    def format_nlu_stage(self, intent: str, confidence: float, entities: List[Dict]) -> str:
        """Format NLU processing stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"🧠 STAGE 1: NLU PROCESSING")
        output.append(f"{'='*80}")
        output.append(f"📍 Intent: {intent} ({confidence:.3f})")
        output.append(f"🏷️  Entities: {len(entities)} found")
        
        for i, entity in enumerate(entities, 1):
            output.append(f"   {i}. {entity['text']} → {entity['type']} ({entity['confidence']:.3f})")
        
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_retrieval_stage(self, policy: str, results_count: int, results: List[Dict] = None) -> str:
        """Format retrieval processing stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"🔍 STAGE 2: INTELLIGENT RETRIEVAL")
        output.append(f"{'='*80}")
        output.append(f"📊 Policy: {policy}")
        output.append(f"📋 Results: {results_count} colleges found")
        
        if results and results_count > 0:
            output.append(f"🎯 Top Matches:")
            for i, result in enumerate(results[:3], 1):
                name = result.get('college_name', 'Unknown')
                score = result.get('confidence', 0)
                output.append(f"   {i}. {name} (score: {score:.3f})")
        
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_context_stage(self, session_id: str, slots: Dict, turn_count: int) -> str:
        """Format context update stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"📝 STAGE 3: CONTEXT UPDATE")
        output.append(f"{'='*80}")
        output.append(f"🔄 Session: {session_id}")
        output.append(f"📊 Slots Updated: {slots}")
        output.append(f"💬 Turn Count: {turn_count}")
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_policy_stage(self, action: str, confidence: float, logic: str) -> str:
        """Format policy planning stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"🤖 STAGE 4: POLICY PLANNING")
        output.append(f"{'='*80}")
        output.append(f"🎬 Selected Action: {action}")
        output.append(f"📈 Intent Confidence: {confidence:.3f}")
        output.append(f"🎯 Decision Logic: {logic}")
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_action_stage(self, action: str, success: bool, retrieval_used: bool, retrieval_count: int = 0) -> str:
        """Format action execution stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"⚡ STAGE 5: ACTION EXECUTION")
        output.append(f"{'='*80}")
        output.append(f"✅ Action: {action}")
        output.append(f"📤 Success: {'✅' if success else '❌'}")
        output.append(f"🔍 Retrieval Used: {'✅' if retrieval_used else '❌'}")
        if retrieval_count > 0:
            output.append(f"📊 Retrieved Results: {retrieval_count} colleges")
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_response_stage(self, response: str, length: int) -> str:
        """Format response generation stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"💬 STAGE 6: RESPONSE GENERATION")
        output.append(f"{'='*80}")
        output.append(f"📝 Response: {response[:100]}...")
        output.append(f"📏 Length: {length} characters")
        output.append(f"{'='*80}")
        return "\n".join(output)
    
    def format_completion_stage(self, session_id: str, action: str, response_length: int, processing_time: float) -> str:
        """Format completion stage"""
        output = []
        output.append(f"\n{'='*80}")
        output.append(f"🎉 DIALOGUE TURN COMPLETED")
        output.append(f"{'='*80}")
        output.append(f"📱 Session: {session_id}")
        output.append(f"🎬 Final Action: {action}")
        output.append(f"💬 Response Sent: {response_length} chars")
        output.append(f"⏱️  Processing Time: {processing_time:.3f}s")
        output.append(f"{'='*80}\n")
        return "\n".join(output)
    
    def parse_log_line(self, log_line: str) -> Dict[str, Any]:
        """Parse a log line to extract stage information"""
        # Extract timestamp
        timestamp_match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})', log_line)
        timestamp = timestamp_match.group(1) if timestamp_match else None
        
        # Extract log level
        level_match = re.search(r' - (\w+) - ', log_line)
        level = level_match.group(1) if level_match else 'INFO'
        
        # Extract message
        message_match = re.search(r' - \w+ - (.+)$', log_line)
        message = message_match.group(1) if message_match else log_line
        
        return {
            'timestamp': timestamp,
            'level': level,
            'message': message,
            'raw': log_line
        }

# Usage example for terminal output
def format_terminal_output(raw_output: str) -> str:
    """Format raw terminal output for better readability"""
    
    formatter = PipelineLogFormatter()
    lines = raw_output.split('\n')
    
    formatted_lines = []
    current_stage = None
    stage_lines = []
    
    for line in lines:
        parsed = formatter.parse_log_line(line)
        message = parsed['message']
        
        # Detect stage transitions
        if "NLU PROCESSING" in message:
            current_stage = "NLU"
        elif "RETRIEVAL" in message:
            current_stage = "RETRIEVAL"
        elif "CONTEXT" in message:
            current_stage = "CONTEXT"
        elif "POLICY" in message:
            current_stage = "POLICY"
        elif "ACTION EXECUTION" in message:
            current_stage = "ACTION"
        elif "RESPONSE GENERATION" in message:
            current_stage = "RESPONSE"
        elif "COMPLETED" in message:
            current_stage = "COMPLETE"
        
        # Apply color formatting
        if current_stage:
            color = formatter.stage_colors.get(current_stage, "")
            reset = formatter.stage_colors["RESET"]
            formatted_line = f"{color}{line}{reset}"
        else:
            formatted_line = line
        
        formatted_lines.append(formatted_line)
    
    return "\n".join(formatted_lines)

if __name__ == "__main__":
    print("🎨 Pipeline Log Formatter")
    print("This module provides enhanced formatting for dialogue pipeline logs.")
    
    # Example usage
    formatter = PipelineLogFormatter()
    
    print(formatter.format_nlu_stage(
        "GET_COLLEGE_INFO", 
        0.85, 
        [{"text": "Kathmandu", "type": "LOCATION", "confidence": 0.95}]
    ))
    
    print(formatter.format_retrieval_stage(
        "semantic_search", 
        3, 
        [
            {"college_name": "KEC", "confidence": 0.92},
            {"college_name": "Pulchowk", "confidence": 0.88}
        ]
    ))