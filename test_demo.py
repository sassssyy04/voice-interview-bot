#!/usr/bin/env python3
"""
Test script to verify voice bot components work correctly.
Run this to test the system without API keys.
"""

import asyncio
import json
from app.models.candidate import Candidate, ShiftPreference, LanguageSkill
from app.models.job import Job, JobCategory
from app.services.job_matching import JobMatchingService
from app.core.logger import logger

async def test_job_matching():
    """Test the job matching algorithm."""
    print("🧪 Testing Job Matching Algorithm...")
    
    # Create a sample candidate
    candidate = Candidate(
        candidate_id="test_001",
        pincode="110001",
        locality="Connaught Place", 
        availability_date="immediately",
        preferred_shift=ShiftPreference.MORNING,
        expected_salary=18000,
        languages=[LanguageSkill.HINDI, LanguageSkill.ENGLISH],
        has_two_wheeler=True,
        total_experience_months=6,
        conversation_completed=True
    )
    
    print(f"📋 Test Candidate Profile:")
    print(f"   Location: {candidate.pincode} ({candidate.locality})")
    print(f"   Salary: ₹{candidate.expected_salary:,}/month")
    print(f"   Shift: {candidate.preferred_shift.value}")
    print(f"   Languages: {[l.value for l in candidate.languages]}")
    print(f"   Two Wheeler: {candidate.has_two_wheeler}")
    print(f"   Experience: {candidate.total_experience_months} months")
    print()
    
    # Test job matching
    matching_service = JobMatchingService()
    matches = await matching_service.find_job_matches(candidate)
    
    print(f"🎯 Found {len(matches.top_matches)} job matches:")
    print(f"   Total jobs considered: {matches.total_jobs_considered}")
    print()
    
    for i, match in enumerate(matches.top_matches, 1):
        score = int(match.match_score * 100)
        print(f"#{i} {match.job.title} at {match.job.company} ({score}% match)")
        print(f"    📍 {match.job.locality} (₹{match.job.salary_min:,}-₹{match.job.salary_max:,})")
        print(f"    💡 {match.rationale}")
        
        if match.strengths:
            print(f"    ✅ Strengths: {'; '.join(match.strengths[:2])}")
        if match.concerns:
            print(f"    ⚠️  Concerns: {'; '.join(match.concerns[:2])}")
        print()
    
    return True

def test_data_models():
    """Test the data models."""
    print("🧪 Testing Data Models...")
    
    # Test candidate model
    candidate = Candidate(
        candidate_id="test_001",
        pincode="110001",
        expected_salary=15000,
        preferred_shift=ShiftPreference.EVENING,
        languages=[LanguageSkill.HINDI, LanguageSkill.ENGLISH],
        has_two_wheeler=True,
        total_experience_months=24
    )
    
    print(f"✅ Candidate model created: {candidate.candidate_id}")
    
    # Test job model
    job = Job(
        job_id="TEST001",
        title="Test Job",
        company="Test Company",
        category=JobCategory.DELIVERY,
        pincode="110001",
        locality="Test Area",
        required_shifts=[ShiftPreference.MORNING],
        salary_min=12000,
        salary_max=18000,
        required_languages=[LanguageSkill.HINDI],
        requires_two_wheeler=False,
        min_experience_months=0,
        description="Test job description",
        contact_number="+91-1234567890"
    )
    
    print(f"✅ Job model created: {job.title}")
    print()
    
    return True

def test_hinglish_prompts():
    """Test Hinglish prompt generation."""
    print("🧪 Testing Hinglish Prompts...")
    
    from app.services.text_to_speech import TTSService
    tts_service = TTSService()
    prompts = tts_service.get_hinglish_prompts()
    
    sample_prompts = [
        "greeting",
        "pincode", 
        "salary",
        "two_wheeler",
        "summary"
    ]
    
    for prompt_key in sample_prompts:
        if prompt_key in prompts:
            prompt_text = prompts[prompt_key]
            if "{" in prompt_text:
                prompt_text = prompt_text.format(value="[sample]")
            print(f"   {prompt_key}: {prompt_text}")
    
    print(f"✅ Generated {len(prompts)} Hinglish prompts")
    print()
    
    return True

def test_conversation_flow():
    """Test conversation flow logic."""
    print("🧪 Testing Conversation Flow...")
    
    from app.services.conversation import ConversationOrchestrator
    orchestrator = ConversationOrchestrator()
    
    # Test conversation flow definition
    flow = orchestrator.conversation_flow
    print(f"   Conversation steps: {' → '.join(flow)}")
    print(f"✅ Conversation flow has {len(flow)} steps")
    print()
    
    return True

async def run_all_tests():
    """Run all test functions."""
    print("🚀 Starting Voice Bot Component Tests\n")
    
    tests = [
        ("Data Models", test_data_models),
        ("Hinglish Prompts", test_hinglish_prompts), 
        ("Conversation Flow", test_conversation_flow),
        ("Job Matching", test_job_matching)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if asyncio.iscoroutinefunction(test_func):
                result = await test_func()
            else:
                result = test_func()
            
            if result:
                print(f"✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"❌ {test_name} - FAILED")
        except Exception as e:
            print(f"❌ {test_name} - ERROR: {e}")
        
        print("-" * 50)
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The voice bot system is ready.")
        print("\n📋 Next Steps:")
        print("1. Set up API keys in .env file")
        print("2. Run: python -m app.main")
        print("3. Open: http://localhost:8000")
        print("4. Click 'Start Interview' and test with voice")
    else:
        print("⚠️  Some tests failed. Check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 