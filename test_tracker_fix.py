import sys
import os
from datetime import datetime

# Add backend to path
sys.path.append(os.path.join(os.getcwd(), "backend"))

from app.context.tracker import DialogueTracker

def test_tracker_structure():
    print("Testing DialogueTracker structure...")
    tracker = DialogueTracker("test_session")
    
    try:
        created_at = tracker.created_at
        print(f"✅ created_at attribute exists: {created_at}")
        assert isinstance(created_at, datetime)
    except AttributeError:
        print("❌ created_at attribute missing!")
    
if __name__ == "__main__":
    test_tracker_structure()
