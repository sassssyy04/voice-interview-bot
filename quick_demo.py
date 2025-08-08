#!/usr/bin/env python3
"""
Quick demo of the Hinglish Voice Bot job matching system.
This runs without voice packages to showcase the core functionality.
"""

from datetime import datetime
import json

# Simple data models without pydantic for demo
class Candidate:
    def __init__(self, candidate_id, pincode=None, expected_salary=None, 
                 preferred_shift=None, languages=None, has_two_wheeler=None, 
                 total_experience_months=None):
        self.candidate_id = candidate_id
        self.pincode = pincode
        self.expected_salary = expected_salary
        self.preferred_shift = preferred_shift
        self.languages = languages or []
        self.has_two_wheeler = has_two_wheeler
        self.total_experience_months = total_experience_months

class Job:
    def __init__(self, job_id, title, company, pincode, locality, 
                 salary_min, salary_max, required_shifts=None, 
                 required_languages=None, requires_two_wheeler=False,
                 min_experience_months=0, contact_number="", description=""):
        self.job_id = job_id
        self.title = title
        self.company = company
        self.pincode = pincode
        self.locality = locality
        self.salary_min = salary_min
        self.salary_max = salary_max
        self.required_shifts = required_shifts or []
        self.required_languages = required_languages or []
        self.requires_two_wheeler = requires_two_wheeler
        self.min_experience_months = min_experience_months
        self.contact_number = contact_number
        self.description = description

# Simple job matching engine
class SimpleJobMatcher:
    def __init__(self):
        self.jobs = [
            Job("DEL001", "Delivery Executive", "QuickDeliver", "110001", "Connaught Place",
                15000, 22000, ["morning", "evening"], ["hindi", "english"], True, 0, "+91-9876543210",
                "Food delivery executive for busy Delhi area"),
            
            Job("SEC002", "Security Guard", "SecureTech", "110002", "Darya Ganj", 
                18000, 25000, ["night"], ["hindi"], False, 6, "+91-9876543211",
                "Night security for commercial building"),
            
            Job("HK003", "Housekeeping Staff", "CleanCorp", "110003", "Civil Lines",
                12000, 18000, ["morning", "afternoon"], ["hindi"], False, 0, "+91-9876543212",
                "Office cleaning and maintenance"),
            
            Job("CON004", "Construction Helper", "BuildRight", "110001", "Connaught Place",
                16000, 24000, ["morning"], ["hindi"], False, 0, "+91-9876543213",
                "Construction site assistance work"),
            
            Job("RET005", "Shop Assistant", "RetailMart", "110002", "Darya Ganj",
                14000, 20000, ["morning", "evening"], ["hindi", "english"], False, 3, "+91-9876543214",
                "Customer service and inventory management")
        ]
    
    def calculate_distance(self, pincode1, pincode2):
        """Simple distance calculation (in real system, use geolocation)"""
        location_map = {
            "110001": (28.6448, 77.2167),  # Connaught Place
            "110002": (28.6469, 77.2167),  # Darya Ganj
            "110003": (28.6505, 77.2324),  # Civil Lines
        }
        
        if pincode1 not in location_map or pincode2 not in location_map:
            return 25  # Default distance
        
        # Simple Euclidean distance approximation
        lat1, lng1 = location_map[pincode1]
        lat2, lng2 = location_map[pincode2]
        
        # Rough conversion to km (simplified)
        distance = ((lat2 - lat1) ** 2 + (lng2 - lng1) ** 2) ** 0.5 * 111
        return distance
    
    def calculate_match_score(self, candidate, job):
        """Calculate match score between candidate and job"""
        scores = {}
        
        # 1. Location Score (30% weight)
        if candidate.pincode and job.pincode:
            distance = self.calculate_distance(candidate.pincode, job.pincode)
            if distance <= 5:
                scores['location'] = 1.0
            elif distance <= 15:
                scores['location'] = 0.8
            elif distance <= 50:
                scores['location'] = 0.5
            else:
                scores['location'] = 0.0
        else:
            scores['location'] = 0.5
        
        # 2. Salary Score (25% weight)
        if candidate.expected_salary:
            if candidate.expected_salary <= job.salary_min:
                scores['salary'] = 1.0
            elif candidate.expected_salary <= job.salary_max:
                range_pos = (candidate.expected_salary - job.salary_min) / (job.salary_max - job.salary_min)
                scores['salary'] = 1.0 - (range_pos * 0.3)
            else:
                overage = (candidate.expected_salary - job.salary_max) / job.salary_max
                scores['salary'] = max(0.0, 0.7 - overage)
        else:
            scores['salary'] = 0.7
        
        # 3. Shift Score (20% weight)
        if candidate.preferred_shift:
            if candidate.preferred_shift in job.required_shifts or candidate.preferred_shift == "flexible":
                scores['shift'] = 1.0
            else:
                scores['shift'] = 0.3
        else:
            scores['shift'] = 0.5
        
        # 4. Language Score (15% weight)
        if candidate.languages:
            matching_langs = set(candidate.languages) & set(job.required_languages)
            if job.required_languages:
                scores['language'] = len(matching_langs) / len(job.required_languages)
            else:
                scores['language'] = 1.0
        else:
            scores['language'] = 0.5
        
        # 5. Vehicle Score (5% weight)
        if job.requires_two_wheeler:
            scores['vehicle'] = 1.0 if candidate.has_two_wheeler else 0.0
        else:
            scores['vehicle'] = 1.0
        
        # 6. Experience Score (5% weight)
        if candidate.total_experience_months is not None:
            if candidate.total_experience_months >= job.min_experience_months:
                scores['experience'] = 1.0
            else:
                gap = (job.min_experience_months - candidate.total_experience_months) / max(job.min_experience_months, 1)
                scores['experience'] = max(0.0, 1.0 - gap)
        else:
            scores['experience'] = 0.5
        
        # Calculate weighted overall score
        weights = {
            'location': 0.30,
            'salary': 0.25,
            'shift': 0.20,
            'language': 0.15,
            'vehicle': 0.05,
            'experience': 0.05
        }
        
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        return overall_score, scores
    
    def generate_rationale(self, candidate, job, score, score_breakdown):
        """Generate human-readable explanation"""
        if score >= 0.8:
            intro = f"Excellent match for {job.title} at {job.company}!"
        elif score >= 0.6:
            intro = f"Good match for {job.title} at {job.company}."
        else:
            intro = f"Partial match for {job.title} at {job.company}."
        
        strengths = []
        concerns = []
        
        if score_breakdown['location'] >= 0.8:
            distance = self.calculate_distance(candidate.pincode, job.pincode)
            strengths.append(f"Job is close ({distance:.1f}km from your location)")
        elif score_breakdown['location'] < 0.5:
            distance = self.calculate_distance(candidate.pincode, job.pincode)
            concerns.append(f"Job is {distance:.1f}km away - consider travel time")
        
        if score_breakdown['salary'] >= 0.8:
            strengths.append(f"Salary range (₹{job.salary_min:,}-₹{job.salary_max:,}) matches your expectations")
        elif score_breakdown['salary'] < 0.7:
            concerns.append(f"Salary may be below your expected ₹{candidate.expected_salary:,}")
        
        if score_breakdown['vehicle'] == 1.0 and job.requires_two_wheeler:
            strengths.append("You have the required two-wheeler")
        elif score_breakdown['vehicle'] == 0.0:
            concerns.append("Job requires two-wheeler which you don't have")
        
        rationale_parts = [intro]
        if strengths:
            rationale_parts.append(f" Strengths: {'; '.join(strengths[:3])}")
        if concerns:
            rationale_parts.append(f" Consider: {'; '.join(concerns[:2])}")
        rationale_parts.append(f" Contact: {job.contact_number}")
        
        return "".join(rationale_parts)
    
    def find_matches(self, candidate):
        """Find top 3 job matches for candidate"""
        job_scores = []
        
        for job in self.jobs:
            score, score_breakdown = self.calculate_match_score(candidate, job)
            if score > 0:
                rationale = self.generate_rationale(candidate, job, score, score_breakdown)
                job_scores.append({
                    'job': job,
                    'score': score,
                    'score_breakdown': score_breakdown,
                    'rationale': rationale
                })
        
        # Sort by score and return top 3
        job_scores.sort(key=lambda x: x['score'], reverse=True)
        return job_scores[:3]

def run_demo():
    """Run the job matching demo"""
    print("🚀 Hinglish Voice Bot - Job Matching Demo")
    print("=" * 50)
    
    # Create sample candidate
    candidate = Candidate(
        candidate_id="demo_001",
        pincode="110001",
        expected_salary=18000,
        preferred_shift="morning",
        languages=["hindi", "english"],
        has_two_wheeler=True,
        total_experience_months=6
    )
    
    print("📋 Sample Candidate Profile:")
    print(f"   Location: {candidate.pincode} (Connaught Place)")
    print(f"   Expected Salary: ₹{candidate.expected_salary:,}/month")
    print(f"   Preferred Shift: {candidate.preferred_shift}")
    print(f"   Languages: {', '.join(candidate.languages)}")
    print(f"   Two Wheeler: {'Yes' if candidate.has_two_wheeler else 'No'}")
    print(f"   Experience: {candidate.total_experience_months} months")
    print()
    
    # Find job matches
    matcher = SimpleJobMatcher()
    matches = matcher.find_matches(candidate)
    
    print("🎯 Top Job Matches:")
    print(f"   Total jobs considered: {len(matcher.jobs)}")
    print(f"   Matches found: {len(matches)}")
    print()
    
    for i, match in enumerate(matches, 1):
        job = match['job']
        score = int(match['score'] * 100)
        
        print(f"#{i} {job.title} at {job.company} ({score}% match)")
        print(f"    📍 {job.locality}")
        print(f"    💰 ₹{job.salary_min:,} - ₹{job.salary_max:,}/month")
        print(f"    📱 {job.contact_number}")
        print(f"    💡 {match['rationale']}")
        
        print(f"    📊 Score Breakdown:")
        for category, score_val in match['score_breakdown'].items():
            percentage = int(score_val * 100)
            print(f"       {category.title()}: {percentage}%")
        print()
    
    print("🔧 How this works:")
    print("1. Location matching (30% weight) - calculates distance between candidate and job")
    print("2. Salary fit (25% weight) - compares expected vs offered salary range")
    print("3. Shift preference (20% weight) - matches work timing preferences") 
    print("4. Language skills (15% weight) - ensures communication requirements")
    print("5. Vehicle requirement (5% weight) - matches transport needs")
    print("6. Experience level (5% weight) - fits qualification requirements")
    print()
    
    print("🎤 Voice Conversation Flow (when voice packages installed):")
    conversation_flow = [
        "Greeting & Consent",
        "Location (Pincode/Area)",
        "Availability Date", 
        "Shift Preference",
        "Expected Salary",
        "Languages Known",
        "Two-wheeler Ownership",
        "Work Experience",
        "Summary & Job Matches"
    ]
    
    for i, step in enumerate(conversation_flow, 1):
        print(f"{i}. {step}")
    
    print()
    print("📱 To enable full voice functionality:")
    print("1. Install voice packages: pip install SpeechRecognition pydub")
    print("2. Get API keys: Azure Speech, Google Cloud Speech, OpenAI")
    print("3. Add keys to .env file")
    print("4. Run: python -m app.main")
    print("5. Open: http://localhost:8000")
    
    print("\n✅ Core job matching system is working!")
    return True

if __name__ == "__main__":
    try:
        run_demo()
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"\n❌ Demo error: {e}")
        import traceback
        traceback.print_exc() 