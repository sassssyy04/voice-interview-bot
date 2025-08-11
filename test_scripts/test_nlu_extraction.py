#!/usr/bin/env python3
"""Test script to verify NLU entity extraction from Devanagari text."""

import asyncio
import sys
import os

# Add the project root to Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.services.nlu import NLUService


async def test_nlu_extraction():
    """Test entity extraction with actual ASR output."""
    
    nlu = NLUService()
    
    print("🧠 Testing NLU Entity Extraction with Devanagari Text...")
    
    # Test cases based on actual ElevenLabs ASR output
    test_cases = [
        {
            "text": "मैं इन दिल्ली में रहता हूं। मेरा पिन कोड है गांव से रॉयटिक।",
            "field": "pincode",
            "expected": "Delhi pincode extraction"
        },
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
        },
        {
            "text": "मैं इमीडिएटली स्टार्ट कर सकता हूं आज से भी।",
            "field": "availability_date",
            "expected": "immediately/today"
        },
        {
            "text": "मैं मॉर्निंग शिफ्ट प्रेफर करता। सुबह का टाइम अच्छा लगता।",
            "field": "preferred_shift",
            "expected": "morning"
        },
        {
            "text": "मैं दो साल से वेयरहाउस में काम कर रहा हूँ।",
            "field": "total_experience_months",
            "expected": "24 months"
        }
    ]
    
    for i, case in enumerate(test_cases, 1):
        print(f"\n--- Test {i}: {case['field']} ---")
        print(f"Input: '{case['text']}'")
        print(f"Expected: {case['expected']}")
        
        try:
            # Test normalization first
            normalized = nlu._normalize_text(case['text'])
            print(f"Normalized: '{normalized}'")
            
            # Test entity extraction
            result = await nlu.extract_entities(case['text'], case['field'])
            
            if result.get('value'):
                print(f"✅ SUCCESS!")
                print(f"   Extracted: {result['value']}")
                print(f"   Confidence: {result.get('confidence', 0):.2f}")
                print(f"   Method: {result.get('method', 'unknown')}")
            else:
                print(f"❌ FAILED - No value extracted")
                print(f"   Result: {result}")
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\n🔍 Summary: NLU entity extraction test completed")


if __name__ == "__main__":
    asyncio.run(test_nlu_extraction()) 