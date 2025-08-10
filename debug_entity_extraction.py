#!/usr/bin/env python3
"""Debug entity extraction and field completion."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.nlu import NLUService
from app.services.conversation import ConversationOrchestrator
from app.models.candidate import Candidate, ConversationState

async def debug_entity_extraction():
    """Test entity extraction and field completion logic."""
    
    print("🔍 Debugging Entity Extraction and Field Completion...")
    
    # Initialize services
    nlu = NLUService()
    orchestrator = ConversationOrchestrator()
    
    # Test cases from our actual test outputs
    test_cases = [
        {
            "text": "भाई, मैं फिफ्टीन थाउजेंड पर मंथ एक्सपेक्ट करता हूं।",
            "field": "expected_salary",
            "expected": "15000"
        },
        {
            "text": "हां भाई, मेरे पास बाइक है बजाज पुलसर।",
            "field": "has_two_wheeler", 
            "expected": "True"
        },
        {
            "text": "मैं हिंदी इंग्लिश दोनों बोल सकता हूं और थोड़ा मराठी भी।",
            "field": "languages",
            "expected": "Hindi, English, Marathi"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {case['field']} ---")
        print(f"Input: '{case['text']}'")
        
        # Step 1: Test NLU extraction
        entities = await nlu.extract_entities(case['text'], case['field'])
        print(f"NLU Result: {entities}")
        
        # Step 2: Test conversation slot update
        candidate = Candidate(candidate_id="test")
        state = ConversationState(candidate_id="test")
        
        print(f"Before update:")
        print(f"  Candidate {case['field']}: {getattr(candidate, case['field'], 'NOT_SET')}")
        print(f"  Fields completed: {state.fields_completed}")
        
        # Test the slot update logic
        updated = orchestrator._try_update_slot(candidate, state, case['field'], entities, case['text'])
        
        print(f"After update:")
        print(f"  Updated returned: {updated}")
        print(f"  Candidate {case['field']}: {getattr(candidate, case['field'], 'NOT_SET')}")
        print(f"  Fields completed: {state.fields_completed}")
        
        if updated:
            print(f"✅ SUCCESS: Field was properly extracted and marked complete")
        else:
            print(f"❌ FAILED: Field was not properly updated")
            
        # Step 3: Test missing slots calculation
        required_slots = ["pincode", "availability_date", "preferred_shift", "expected_salary", 
                         "languages", "has_two_wheeler", "total_experience_months", "confirmation"]
        completed_slots = set(state.fields_completed)
        missing_slots = [s for s in required_slots if s not in completed_slots]
        
        print(f"  Missing slots after update: {missing_slots}")
        print(f"  Is {case['field']} still missing? {case['field'] in missing_slots}")

if __name__ == "__main__":
    asyncio.run(debug_entity_extraction()) 