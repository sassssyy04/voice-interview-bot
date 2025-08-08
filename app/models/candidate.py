from pydantic import BaseModel, Field
from typing import Optional, List
from datetime import datetime
from enum import Enum


class ShiftPreference(str, Enum):
    MORNING = "morning"
    AFTERNOON = "afternoon"
    EVENING = "evening" 
    NIGHT = "night"
    FLEXIBLE = "flexible"


class LanguageSkill(str, Enum):
    HINDI = "hindi"
    ENGLISH = "english"
    MARATHI = "marathi"
    BENGALI = "bengali"
    TAMIL = "tamil"
    TELUGU = "telugu"
    GUJARATI = "gujarati"
    KANNADA = "kannada"
    PUNJABI = "punjabi"


class Candidate(BaseModel):
    """Candidate profile collected through voice conversation."""
    
    # Basic info
    candidate_id: str = Field(..., description="Unique candidate identifier")
    created_at: datetime = Field(default_factory=datetime.now)
    
    # Location
    pincode: Optional[str] = Field(None, description="6-digit pincode")
    locality: Optional[str] = Field(None, description="Area/locality name")
    
    # Availability
    availability_date: Optional[str] = Field(None, description="When can start work")
    preferred_shift: Optional[ShiftPreference] = Field(None, description="Preferred work shift")
    
    # Compensation
    expected_salary: Optional[int] = Field(None, description="Expected monthly salary in INR")
    
    # Skills & Requirements
    languages: List[LanguageSkill] = Field(default_factory=list, description="Known languages")
    has_two_wheeler: Optional[bool] = Field(None, description="Owns 2-wheeler")
    total_experience_months: Optional[int] = Field(None, description="Total work experience in months")
    
    # Conversation metadata
    conversation_completed: bool = Field(False, description="Whether screening is complete")
    turn_count: int = Field(0, description="Number of conversation turns")
    

class ConversationState(BaseModel):
    """Current state of the voice conversation."""
    
    candidate_id: str
    current_field: str = Field("greeting", description="Current field being collected")
    current_step: int = Field(0, description="Index in the conversation flow")
    fields_completed: List[str] = Field(default_factory=list)
    retry_count: int = Field(0, description="Retries for current field")
    last_confidence: float = Field(0.0, description="Last ASR confidence score")
    needs_confirmation: bool = Field(False, description="Whether last input needs confirmation")
    pending_confirmation_value: Optional[str] = Field(None, description="Value awaiting confirmation")
    
    @property
    def completion_rate(self) -> float:
        """Calculate percentage of required fields completed."""
        required_fields = [
            "pincode", "availability_date", "preferred_shift", 
            "expected_salary", "languages", "has_two_wheeler", "total_experience_months"
        ]
        completed = len([f for f in required_fields if f in self.fields_completed])
        return completed / len(required_fields)


class VoiceTurn(BaseModel):
    """Single turn in voice conversation with telemetry."""
    
    turn_id: str
    candidate_id: str
    timestamp: datetime = Field(default_factory=datetime.now)
    
    # Input
    asr_text: str = Field("", description="Recognized speech text")
    asr_confidence: float = Field(0.0, description="ASR confidence score")
    
    # Processing
    extracted_entities: dict = Field(default_factory=dict, description="NLU extracted entities")
    chosen_prompt: str = Field("", description="TTS prompt used")
    
    # Output
    tts_text: str = Field("", description="Text sent to TTS")
    tts_char_count: int = Field(0, description="Number of characters in TTS")
    
    # Performance
    asr_latency_ms: float = Field(0.0, description="ASR processing time")
    nlu_latency_ms: float = Field(0.0, description="NLU processing time") 
    tts_latency_ms: float = Field(0.0, description="TTS processing time")
    total_latency_ms: float = Field(0.0, description="Total turn processing time") 