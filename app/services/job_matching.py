import json
import math
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from app.models.candidate import Candidate
from app.models.job import Job, JobMatch, MatchingResult
from app.core.config import settings
from app.core.logger import logger

try:
    import openai
except ImportError:
    openai = None


class JobMatchingService:
    """Job matching service with transparent scoring and rationales."""
    
    def __init__(self):
        self.location_data = self._load_location_data()
        if openai and settings.openai_api_key:
            openai.api_key = settings.openai_api_key
    
    def _load_location_data(self) -> Dict[str, Dict[str, float]]:
        """Load pincode to location mapping with coordinates."""
        # Sample location data - in real implementation, load from database
        return {
            "110001": {"lat": 28.6448, "lng": 77.2167, "name": "Connaught Place"},
            "110002": {"lat": 28.6469, "lng": 77.2167, "name": "Darya Ganj"},
            "110003": {"lat": 28.6505, "lng": 77.2324, "name": "Civil Lines"},
            "400001": {"lat": 18.9322, "lng": 72.8264, "name": "Fort Mumbai"},
            "400002": {"lat": 18.9467, "lng": 72.8073, "name": "Kalbadevi"},
            "400003": {"lat": 18.9481, "lng": 72.8347, "name": "Masjid"},
            "560001": {"lat": 12.9716, "lng": 77.5946, "name": "Bangalore GPO"},
            "560002": {"lat": 12.9698, "lng": 77.6044, "name": "Bangalore City"},
            "700001": {"lat": 22.5726, "lng": 88.3639, "name": "Kolkata GPO"}
        }
    
    async def find_job_matches(self, candidate: Candidate) -> MatchingResult:
        """Find top 3 job matches for a candidate.
        
        Args:
            candidate (Candidate): Candidate profile
            
        Returns:
            MatchingResult: Top matches with rationales
        """
        logger.info(f"Finding job matches for candidate {candidate.candidate_id}")
        
        # Load available jobs
        jobs = await self._load_available_jobs()
        
        # Calculate match scores for all jobs with 60km distance filtering
        job_scores = []
        for job in jobs:
            # Apply 60km distance filter before calculating full match score
            if candidate.pincode and job.pincode:
                distance_km = self._calculate_distance(candidate.pincode, job.pincode)
                if distance_km > 60.0:
                    continue  # Skip jobs beyond 60km
            
            match = self._calculate_job_match(candidate, job)
            if match.match_score > 0:  # Only include viable matches
                job_scores.append(match)
        
        # Sort by match score
        job_scores.sort(key=lambda x: x.match_score, reverse=True)
        
        # Take top 3 matches
        top_matches = job_scores[:3]
        
        # If no matches found within distance, log it
        if len(job_scores) == 0:
            logger.info(f"No jobs found within 60km for candidate {candidate.candidate_id} at pincode {candidate.pincode}")
        
        result = MatchingResult(
            candidate_id=candidate.candidate_id,
            top_matches=top_matches,
            total_jobs_considered=len(jobs),
            matching_criteria_used=self._get_matching_criteria(),
            generated_at=datetime.now().isoformat()
        )
        
        logger.bind(metrics=True).info({
            "event": "job_matching_completed",
            "candidate_id": candidate.candidate_id,
            "total_jobs": len(jobs),
            "matches_found": len(top_matches),
            "top_score": top_matches[0].match_score if top_matches else 0
        })
        
        return result
    
    async def generate_personalized_summary(self, candidate: Candidate, result: MatchingResult) -> str:
        """Generate a customized Hinglish recommendation using an LLM if available.
        
        Args:
            candidate: Candidate profile
            result: MatchingResult with top_matches
        Returns:
            str: Short Hinglish summary tailored to the candidate
        """
        # Build concise structured context
        def match_to_dict(m: JobMatch) -> Dict[str, object]:
            return {
                "title": m.job.title,
                "company": m.job.company,
                "locality": m.job.locality,
                "pincode": m.job.pincode,
                "salary_min": m.job.salary_min,
                "salary_max": m.job.salary_max,
                "shifts": [s.value for s in m.job.required_shifts],
                "languages": [l.value for l in m.job.required_languages],
                "requires_two_wheeler": m.job.requires_two_wheeler,
                "match_score": round(m.match_score, 3),
                "strengths": m.strengths[:3],
                "concerns": m.concerns[:3],
            }
        
        top = [match_to_dict(m) for m in result.top_matches]
        cand = {
            "pincode": candidate.pincode,
            "locality": candidate.locality,
            "availability": candidate.availability_date,
            "preferred_shift": candidate.preferred_shift.value if candidate.preferred_shift else None,
            "expected_salary": candidate.expected_salary,
            "languages": [l.value for l in candidate.languages] if candidate.languages else [],
            "has_two_wheeler": candidate.has_two_wheeler,
            "experience_months": candidate.total_experience_months,
        }
        
        system_prompt = (
            "You are a recruitment voice assistant for Indian blue-collar workers. "
            "Speak Hinglish (simple Hindi and english where required), short lines, neutral Indian tone. "
            "Keep under 100-120 words, no long paragraphs, easy to understand. "
            "Explain top 3 jobs in plain language with why it's a fit (pay, distance/location, shift, language, 2-wheeler, experience). "
            "Be honest about concerns if any. End with a short question asking which job they want to hear more about."
        )
        
        # Handle no matches case for LLM
        if not result.top_matches:
            user_prompt = (
                "No job matches found within 60km radius for the candidate. "
                "Return a respectful Hinglish message explaining no suitable jobs are available in their area "
                "and that they will be notified when matching jobs become available."
            )
        else:
            user_prompt = (
                "Candidate profile and job matches are below as JSON. "
                "Return a single Hinglish paragraph (2-3 lines) plus a short bullet list with 1 line per job. "
                "Avoid technical words. Use rupee symbol. Keep respectful tone.\n\n"
                f"Candidate: {json.dumps(cand, ensure_ascii=False)}\n"
                f"TopMatches: {json.dumps(top, ensure_ascii=False)}"
            )
        
        # If OpenAI unavailable, fallback to a compact rule-based summary
        if not (openai and settings.openai_api_key):
            # Handle no jobs case
            if not result.top_matches:
                return "Maaf kijiye, aapke area mein 60km ke radius mein koi suitable job available nahi hai abhi. Hum aapko notify kar denge jab koi matching job mil jayegi."
            
            parts = ["Aapke details dekhkar ye jobs best lag rahi hain:"]
            bullets = []
            for i, m in enumerate(result.top_matches[:3], 1):
                bullets.append(
                    f"{i}) {m.job.title} - {m.job.company} ({m.job.locality}) • ₹{m.job.salary_min:,}-₹{m.job.salary_max:,}. "
                    f"Reason: {m.rationale}"
                )
            parts.extend(bullets)
            parts.append("Kaunsi job pe aap interest rakhte ho? Main details bata deta/ deti hoon.")
            return "\n".join(parts)
        
        try:
            response = await self._llm_call(system_prompt, user_prompt)
            text = response.strip()
            # Safety trim
            if len(text) > 800:
                text = text[:800]
            return text
        except Exception as e:
            logger.error(f"LLM personalization failed: {e}")
            # Fallback to rule-based summary
            if not result.top_matches:
                return "Maaf kijiye, aapke area mein 60km ke radius mein koi suitable job available nahi hai abhi. Hum aapko notify kar denge jab koi matching job mil jayegi."
            
            parts = ["Aapke details dekhkar ye jobs best lag rahi hain:"]
            bullets = []
            for i, m in enumerate(result.top_matches[:3], 1):
                bullets.append(
                    f"{i}) {m.job.title} - {m.job.company} ({m.job.locality}) • ₹{m.job.salary_min:,}-₹{m.job.salary_max:,}. "
                    f"Reason: {m.rationale}"
                )
            parts.extend(bullets)
            parts.append("Kaunsi job pe aap interest rakhte ho? Main details bata deta/ deti hoon.")
            return "\n".join(parts)
    
    async def _llm_call(self, system_prompt: str, user_prompt: str) -> str:
        """Call OpenAI chat completion asynchronously using a thread."""
        if not (openai and settings.openai_api_key):
            raise RuntimeError("OpenAI not configured")
        import asyncio
        return (await asyncio.to_thread(
            openai.chat.completions.create,
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=220,
            temperature=0.3,
        )).choices[0].message.content

    def _calculate_job_match(self, candidate: Candidate, job: Job) -> JobMatch:
        """Calculate detailed match score between candidate and job.
        
        Args:
            candidate (Candidate): Candidate profile
            job (Job): Job listing
            
        Returns:
            JobMatch: Match details with scoring breakdown
        """
        # Initialize scores
        location_score = 0.0
        salary_score = 0.0
        shift_score = 0.0
        language_score = 0.0
        vehicle_score = 0.0
        experience_score = 0.0
        
        concerns = []
        strengths = []
        
        # 1. Location Score (30% weight)
        if candidate.pincode and job.pincode:
            distance_km = self._calculate_distance(candidate.pincode, job.pincode)
            if distance_km <= 5:
                location_score = 1.0
                strengths.append(f"Job is very close ({distance_km:.1f}km from your location)")
            elif distance_km <= 15:
                location_score = 0.8
                strengths.append(f"Job is nearby ({distance_km:.1f}km from your location)")
            elif distance_km <= settings.max_distance_km:
                location_score = 0.5
                concerns.append(f"Job is {distance_km:.1f}km away - consider travel time")
            else:
                location_score = 0.0
                concerns.append(f"Job is too far ({distance_km:.1f}km) - may not be practical")
        elif candidate.locality and job.locality:
            # Light boost for same locality when pincode missing
            if candidate.locality.strip().lower() == job.locality.strip().lower():
                location_score = 0.7
                strengths.append("Same locality as your area")
            else:
                location_score = 0.4
                concerns.append("Pincode missing; using locality approximation")
        
        # 2. Salary Score (25% weight)
        if candidate.expected_salary:
            salary_fit = self._calculate_salary_fit(candidate.expected_salary, job.salary_min, job.salary_max)
            salary_score = salary_fit
            
            if salary_fit >= 0.9:
                strengths.append(f"Salary range (₹{job.salary_min:,}-₹{job.salary_max:,}) matches your expectations")
            elif salary_fit >= 0.7:
                strengths.append(f"Salary is close to your expectations")
            else:
                concerns.append(f"Salary (₹{job.salary_min:,}-₹{job.salary_max:,}) is below your expected ₹{candidate.expected_salary:,}")
        
        # 3. Shift Score (20% weight)
        if candidate.preferred_shift:
            if candidate.preferred_shift in job.required_shifts or candidate.preferred_shift.value == "flexible":
                shift_score = 1.0
                strengths.append(f"{candidate.preferred_shift.value.title()} shift is available")
            else:
                shift_score = 0.3
                available_shifts = [s.value for s in job.required_shifts]
                concerns.append(f"Your preferred {candidate.preferred_shift.value} shift not available. Available: {', '.join(available_shifts)}")
        
        # 4. Language Score (15% weight)
        if candidate.languages and job.required_languages:
            language_match = len(set(candidate.languages) & set(job.required_languages))
            language_total = len(job.required_languages)
            language_score = language_match / language_total if language_total > 0 else 1.0
            
            if language_score >= 0.8:
                strengths.append(f"You speak the required languages ({', '.join([l.value for l in candidate.languages])})")
            else:
                missing_langs = set(job.required_languages) - set(candidate.languages)
                concerns.append(f"Missing required languages: {', '.join([l.value for l in missing_langs])}")
        
        # 5. Vehicle Score (5% weight)
        if job.requires_two_wheeler:
            if candidate.has_two_wheeler:
                vehicle_score = 1.0
                strengths.append("You have the required two-wheeler")
            else:
                vehicle_score = 0.0
                concerns.append("Job requires two-wheeler which you don't have")
        else:
            vehicle_score = 1.0  # No vehicle required
        
        # 6. Experience Score (5% weight)
        if candidate.total_experience_months is not None:
            exp_fit = self._calculate_experience_fit(
                candidate.total_experience_months,
                job.min_experience_months,
                job.max_experience_months
            )
            experience_score = exp_fit
            
            if exp_fit >= 0.9:
                strengths.append(f"Your {candidate.total_experience_months} months experience fits perfectly")
            elif exp_fit >= 0.7:
                strengths.append("Your experience level is suitable")
            elif candidate.total_experience_months < job.min_experience_months:
                concerns.append(f"Job requires {job.min_experience_months} months experience, you have {candidate.total_experience_months}")
            else:
                concerns.append("You may be overqualified for this role")
        
        # Calculate weighted overall score
        weights = {
            "location": 0.30,
            "salary": 0.25,
            "shift": 0.20,
            "language": 0.15,
            "vehicle": 0.05,
            "experience": 0.05
        }
        
        overall_score = (
            location_score * weights["location"] +
            salary_score * weights["salary"] +
            shift_score * weights["shift"] +
            language_score * weights["language"] +
            vehicle_score * weights["vehicle"] +
            experience_score * weights["experience"]
        )
        
        # Generate human-readable rationale
        rationale = self._generate_rationale(job, overall_score, strengths, concerns)
        
        return JobMatch(
            job=job,
            match_score=overall_score,
            location_score=location_score,
            salary_score=salary_score,
            shift_score=shift_score,
            language_score=language_score,
            vehicle_score=vehicle_score,
            experience_score=experience_score,
            rationale=rationale,
            concerns=concerns,
            strengths=strengths
        )
    
    def _calculate_distance(self, pincode1: str, pincode2: str) -> float:
        """Calculate distance between two pincodes using simple pincode difference.
        
        Uses direct subtraction of pincode numbers as a proxy for distance.
        Every 1000 pincode difference ≈ 100km approximately.
        """
        try:
            if not pincode1 or not pincode2:
                return 100.0  # Default large distance for missing pincodes
            
            # Convert pincodes to integers for direct subtraction
            pin1 = int(pincode1)
            pin2 = int(pincode2)
            
            # Calculate absolute difference
            pincode_diff = abs(pin1 - pin2)
            
            # Convert pincode difference to approximate kilometers
            # 1000 pincode difference ≈ 100km (rough approximation for India)
            distance_km = pincode_diff / 10  # Every 10 pincode units = 1km
            
            # Cap maximum distance for very different pincodes
            return min(distance_km, 500.0)
            
        except (ValueError, TypeError):
            logger.error(f"Error calculating pincode distance: {pincode1} vs {pincode2}")
            return 100.0  # Default large distance for invalid pincodes
    
    def _calculate_salary_fit(self, expected: int, job_min: int, job_max: int) -> float:
        """Calculate how well job salary matches candidate expectation."""
        if expected <= job_min:
            return 1.0  # Candidate expects less than minimum - perfect fit
        elif expected <= job_max:
            # Linear interpolation within range
            range_position = (expected - job_min) / (job_max - job_min)
            return 1.0 - (range_position * 0.3)  # Slight penalty for higher expectations
        else:
            # Expectation exceeds maximum
            overage = (expected - job_max) / job_max
            tolerance = settings.salary_tolerance_percent / 100
            
            if overage <= tolerance:
                return 0.7  # Within tolerance
            else:
                return max(0.0, 0.7 - (overage - tolerance))  # Decreasing score
    
    def _calculate_experience_fit(self, candidate_exp: int, min_req: int, max_req: Optional[int]) -> float:
        """Calculate experience fit score."""
        if candidate_exp < min_req:
            # Under-qualified
            gap = (min_req - candidate_exp) / max(min_req, 1)
            return max(0.0, 1.0 - gap)
        elif max_req is None or candidate_exp <= max_req:
            # Perfect fit
            return 1.0
        else:
            # Over-qualified
            excess = (candidate_exp - max_req) / max_req
            return max(0.3, 1.0 - (excess * 0.5))  # Gentle penalty for overqualification
    
    def _generate_rationale(self, job: Job, score: float, strengths: List[str], concerns: List[str]) -> str:
        """Generate human-readable rationale for the match."""
        if score >= 0.8:
            intro = f"Excellent match for {job.title} at {job.company}!"
        elif score >= 0.6:
            intro = f"Good match for {job.title} at {job.company}."
        else:
            intro = f"Partial match for {job.title} at {job.company}."
        
        rationale_parts = [intro]
        
        if strengths:
            rationale_parts.append(" Key strengths: " + "; ".join(strengths[:3]))
        
        if concerns:
            rationale_parts.append(" Consider: " + "; ".join(concerns[:2]))
        
        rationale_parts.append(f" Contact: {job.contact_number}")
        
        return "".join(rationale_parts)
    
    def _get_matching_criteria(self) -> Dict[str, float]:
        """Get the criteria weights used for matching."""
        return {
            "location_weight": 0.30,
            "salary_weight": 0.25,
            "shift_weight": 0.20,
            "language_weight": 0.15,
            "vehicle_weight": 0.05,
            "experience_weight": 0.05,
            "max_distance_km": settings.max_distance_km,
            "salary_tolerance_percent": settings.salary_tolerance_percent
        }
    
    async def _load_available_jobs(self) -> List[Job]:
        """Load available jobs from data source."""
        # In real implementation, load from database
        # For demo, return sample jobs
        return [
            Job(
                job_id="DEL001",
                title="Delivery Executive",
                company="QuickDeliver",
                category="delivery",
                pincode="110001",
                locality="Connaught Place",
                required_shifts=["morning", "evening"],
                salary_min=15000,
                salary_max=22000,
                required_languages=["hindi", "english"],
                requires_two_wheeler=True,
                min_experience_months=0,
                max_experience_months=24,
                description="Food delivery executive for busy Delhi area",
                benefits=["Fuel allowance", "Mobile phone", "Incentives"],
                contact_number="+91-9876543210"
            ),
            Job(
                job_id="SEC002",
                title="Security Guard",
                company="SecureTech",
                category="security",
                pincode="110002",
                locality="Darya Ganj",
                required_shifts=["night"],
                salary_min=18000,
                salary_max=25000,
                required_languages=["hindi"],
                requires_two_wheeler=False,
                min_experience_months=6,
                max_experience_months=60,
                description="Night security for commercial building",
                benefits=["Medical insurance", "Uniform provided"],
                contact_number="+91-9876543211"
            ),
            Job(
                job_id="HK003",
                title="Housekeeping Staff",
                company="CleanCorp",
                category="housekeeping",
                pincode="110003",
                locality="Civil Lines",
                required_shifts=["morning", "afternoon"],
                salary_min=12000,
                salary_max=18000,
                required_languages=["hindi"],
                requires_two_wheeler=False,
                min_experience_months=0,
                max_experience_months=36,
                description="Office cleaning and maintenance",
                benefits=["Weekly off", "Festival bonus"],
                contact_number="+91-9876543212"
            ),
            Job(
                job_id="CON004",
                title="Construction Helper",
                company="BuildRight",
                category="construction",
                pincode="110001",
                locality="Connaught Place",
                required_shifts=["morning"],
                salary_min=16000,
                salary_max=24000,
                required_languages=["hindi"],
                requires_two_wheeler=False,
                min_experience_months=0,
                max_experience_months=24,
                description="Construction site assistance work",
                benefits=["Safety equipment", "Overtime pay"],
                contact_number="+91-9876543213"
            ),
            Job(
                job_id="RET005",
                title="Shop Assistant",
                company="RetailMart",
                category="retail",
                pincode="110002",
                locality="Darya Ganj",
                required_shifts=["morning", "evening"],
                salary_min=14000,
                salary_max=20000,
                required_languages=["hindi", "english"],
                requires_two_wheeler=False,
                min_experience_months=3,
                max_experience_months=48,
                description="Customer service and inventory management",
                benefits=["Staff discount", "Training provided"],
                contact_number="+91-9876543214"
            )
        ] 