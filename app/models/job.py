from pydantic import BaseModel, Field
from typing import List, Optional
from enum import Enum
from app.models.candidate import ShiftPreference, LanguageSkill


class JobCategory(str, Enum):
    DELIVERY = "delivery"
    SECURITY = "security"
    HOUSEKEEPING = "housekeeping"
    CONSTRUCTION = "construction"
    MANUFACTURING = "manufacturing"
    RETAIL = "retail"
    FOOD_SERVICE = "food_service"
    TRANSPORTATION = "transportation"


class Job(BaseModel):
    """Job listing with requirements and details."""
    
    job_id: str = Field(..., description="Unique job identifier")
    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    category: JobCategory = Field(..., description="Job category")
    
    # Location
    pincode: str = Field(..., description="Job location pincode")
    locality: str = Field(..., description="Job location area")
    
    # Requirements
    required_shifts: List[ShiftPreference] = Field(..., description="Available work shifts")
    salary_min: int = Field(..., description="Minimum monthly salary in INR")
    salary_max: int = Field(..., description="Maximum monthly salary in INR")
    required_languages: List[LanguageSkill] = Field(..., description="Required languages")
    requires_two_wheeler: bool = Field(False, description="Whether 2-wheeler is mandatory")
    min_experience_months: int = Field(0, description="Minimum experience required in months")
    max_experience_months: Optional[int] = Field(None, description="Maximum experience in months")
    
    # Job details
    description: str = Field(..., description="Job description")
    benefits: List[str] = Field(default_factory=list, description="Job benefits")
    contact_number: str = Field(..., description="Contact number for applications")
    
    # Internal
    is_active: bool = Field(True, description="Whether job is currently active")


class JobMatch(BaseModel):
    """Job match result with scoring details."""
    
    job: Job
    match_score: float = Field(..., description="Overall match score (0-1)")
    
    # Detailed scoring
    location_score: float = Field(..., description="Location proximity score")
    salary_score: float = Field(..., description="Salary fit score")
    shift_score: float = Field(..., description="Shift preference score")
    language_score: float = Field(..., description="Language requirement score")
    vehicle_score: float = Field(..., description="2-wheeler requirement score")
    experience_score: float = Field(..., description="Experience level score")
    
    # Explanations
    rationale: str = Field(..., description="Human-readable explanation of the match")
    concerns: List[str] = Field(default_factory=list, description="Potential issues with the match")
    strengths: List[str] = Field(default_factory=list, description="Why this is a good match")


class MatchingResult(BaseModel):
    """Complete matching result for a candidate."""
    
    candidate_id: str
    top_matches: List[JobMatch] = Field(..., description="Top 3 job matches")
    total_jobs_considered: int = Field(..., description="Total number of jobs evaluated")
    matching_criteria_used: dict = Field(..., description="Criteria used for matching")
    generated_at: str = Field(..., description="When results were generated") 