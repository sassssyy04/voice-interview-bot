#!/usr/bin/env python3
"""
Simplified test script that works without voice packages.
Tests core functionality like job matching and data models.
"""

import asyncio
import sys
import os

def test_basic_imports():
    """Test that basic imports work."""
    print("🧪 Testing Basic Imports...")
    
    try:
        from app.models.candidate import Candidate, ShiftPreference, LanguageSkill
        print("✅ Candidate models imported successfully")
        
        from app.models.job import Job, JobCategory
        print("✅ Job models imported successfully")
        
        from app.core.config import settings
        print("✅ Configuration imported successfully")
        
        from app.core.logger import logger
        print("✅ Logger imported successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_data_models():
    """Test the data models work correctly."""
    print("\n🧪 Testing Data Models...")
    
    try:
        from app.models.candidate import Candidate, ShiftPreference, LanguageSkill
        from app.models.job import Job, JobCategory
        
        # Test candidate creation
        candidate = Candidate(
            candidate_id="test_001",
            pincode="110001",
            expected_salary=15000,
            preferred_shift=ShiftPreference.EVENING,
            languages=[LanguageSkill.HINDI, LanguageSkill.ENGLISH],
            has_two_wheeler=True,
            total_experience_months=24
        )
        
        print(f"✅ Candidate created: {candidate.candidate_id}")
        print(f"   Location: {candidate.pincode}")
        print(f"   Salary: ₹{candidate.expected_salary:,}")
        print(f"   Shift: {candidate.preferred_shift.value}")
        
        # Test job creation
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
        
        print(f"✅ Job created: {job.title} at {job.company}")
        print(f"   Salary: ₹{job.salary_min:,} - ₹{job.salary_max:,}")
        
        return True
    except Exception as e:
        print(f"❌ Data model test failed: {e}")
        return False

async def test_job_matching():
    """Test the job matching algorithm."""
    print("\n🧪 Testing Job Matching Algorithm...")
    
    try:
        from app.models.candidate import Candidate, ShiftPreference, LanguageSkill
        from app.services.job_matching import JobMatchingService
        
        # Create test candidate
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
        
        print(f"📋 Test Candidate:")
        print(f"   Location: {candidate.pincode}")
        print(f"   Salary: ₹{candidate.expected_salary:,}")
        print(f"   Languages: {[l.value for l in candidate.languages]}")
        print(f"   Two Wheeler: {candidate.has_two_wheeler}")
        
        # Test job matching
        matching_service = JobMatchingService()
        matches = await matching_service.find_job_matches(candidate)
        
        print(f"\n🎯 Job Matching Results:")
        print(f"   Total jobs considered: {matches.total_jobs_considered}")
        print(f"   Matches found: {len(matches.top_matches)}")
        
        for i, match in enumerate(matches.top_matches, 1):
            score = int(match.match_score * 100)
            job = match.job
            print(f"\n#{i} {job.title} at {job.company} ({score}% match)")
            print(f"    📍 {job.locality}")
            print(f"    💰 ₹{job.salary_min:,} - ₹{job.salary_max:,}")
            print(f"    📱 {job.contact_number}")
            print(f"    💡 {match.rationale}")
            
            if match.strengths:
                print(f"    ✅ Strengths: {'; '.join(match.strengths[:2])}")
            if match.concerns:
                print(f"    ⚠️  Concerns: {'; '.join(match.concerns[:2])}")
        
        return True
    except Exception as e:
        print(f"❌ Job matching test failed: {e}")
        return False

def test_web_server():
    """Test that the web server can be imported."""
    print("\n🧪 Testing Web Server Components...")
    
    try:
        from app.main import app
        print("✅ FastAPI app imported successfully")
        
        from app.api.routes import router
        print("✅ API routes imported successfully")
        
        print("✅ Web server components ready")
        print("   You can start the server with: python -m app.main")
        
        return True
    except Exception as e:
        print(f"❌ Web server test failed: {e}")
        return False

async def run_all_tests():
    """Run all available tests."""
    print("🚀 Running Simplified Voice Bot Tests\n")
    print("=" * 50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Data Models", test_data_models),
        ("Job Matching", test_job_matching),
        ("Web Server", test_web_server)
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
                print(f"\n✅ {test_name} - PASSED")
                passed += 1
            else:
                print(f"\n❌ {test_name} - FAILED")
        except Exception as e:
            print(f"\n❌ {test_name} - ERROR: {e}")
            import traceback
            traceback.print_exc()
        
        print("-" * 50)
    
    print(f"\n🏁 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All core tests passed! The system is ready.")
        print("\n📋 Next Steps:")
        print("1. Set up API keys in .env file (copy from env_example.txt)")
        print("2. Start the server: python -m app.main") 
        print("3. Open: http://localhost:8000")
        print("4. Test the web interface")
        print("\n🔧 For full voice functionality, install:")
        print("   pip install SpeechRecognition pydub openai")
        print("   pip install google-cloud-speech google-cloud-texttospeech")
        print("   pip install azure-cognitiveservices-speech")
        
    else:
        print(f"\n⚠️ {total - passed} tests failed.")
        print("Check the errors above and ensure all packages are installed.")
        print("\nTo install missing packages:")
        print("   python setup.py")
    
    return passed == total

if __name__ == "__main__":
    asyncio.run(run_all_tests()) 