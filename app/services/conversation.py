import asyncio
import uuid
import time
from typing import Dict, Optional, Tuple, Any
from datetime import datetime

from app.models.candidate import Candidate, ConversationState, VoiceTurn, ShiftPreference, LanguageSkill
from app.services.speech_recognition import ASRService
from app.services.text_to_speech import TTSService
from app.services.nlu import NLUService
from app.services.job_matching import JobMatchingService
from app.core.config import settings
from app.core.logger import logger


class ConversationOrchestrator:
    """Orchestrates the voice conversation flow for candidate screening."""
    
    def __init__(self):
        try:
            self.asr_service = ASRService()
            self.tts_service = TTSService()
            self.nlu_service = NLUService()
        except ImportError as e:
            logger.warning(f"Voice services not available: {e}")
            self.asr_service = None
            self.tts_service = None
            self.nlu_service = None
        
        self.job_matching_service = JobMatchingService()
        
        # In-memory storage for demo (use database in production)
        self.candidates: Dict[str, Candidate] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        self.conversation_turns: Dict[str, list] = {}
        
        # Conversation flow definition
        self.conversation_flow = [
            "greeting",
            "pincode", 
            "availability_date",
            "preferred_shift",
            "expected_salary",
            "languages",
            "has_two_wheeler", 
            "total_experience_months",
            "summary"
        ]
        
        # Only get prompts if TTS service is available
        if self.tts_service:
            self.prompts = self.tts_service.get_hinglish_prompts()
        else:
            self.prompts = {}
        
        # Mapping from state field to prompt key in TTS prompts
        self.prompt_key_map = {
            "greeting": "greeting",
            "pincode": "pincode",
            "availability_date": "availability",
            "preferred_shift": "shift",
            "expected_salary": "salary",
            "languages": "languages",
            "has_two_wheeler": "two_wheeler",
            "total_experience_months": "experience",
            "summary": "summary",
        }
    
    async def start_conversation(self) -> Tuple[str, bytes]:
        """Start a new conversation session.
        
        Returns:
            Tuple[str, bytes]: (candidate_id, initial_prompt_audio)
        """
        if self.tts_service is None:
            raise Exception("Voice services not available - Google Cloud APIs need to be enabled")
            
        candidate_id = str(uuid.uuid4())
        
        # Initialize candidate
        candidate = Candidate(candidate_id=candidate_id)
        self.candidates[candidate_id] = candidate
        
        # Initialize conversation state
        state = ConversationState(candidate_id=candidate_id)
        self.conversation_states[candidate_id] = state
        self.conversation_turns[candidate_id] = []
        
        # Generate initial greeting
        greeting_text = self.prompts["greeting"]
        audio_data = await self.tts_service.synthesize_speech(greeting_text)
        
        # Log conversation start
        logger.bind(conversation=True).info({
            "event": "conversation_started",
            "candidate_id": candidate_id,
            "initial_prompt": greeting_text
        })
        
        return candidate_id, audio_data
    
    async def process_turn(self, candidate_id: str, audio_data: bytes) -> Tuple[str, bytes, bool]:
        """Process a voice turn from the candidate.
        
        Args:
            candidate_id: The candidate's unique ID
            audio_data: Raw audio bytes from the candidate
            
        Returns:
            Tuple[str, bytes, bool]: (response_text, response_audio, is_completed)
        """
        if self.asr_service is None or self.tts_service is None or self.nlu_service is None:
            raise Exception("Voice services not available - Google Cloud APIs need to be enabled")
            
        start_time = time.time()
        
        # Get conversation state
        state = self.conversation_states.get(candidate_id)
        candidate = self.candidates.get(candidate_id)
        
        if not state or not candidate:
            raise ValueError(f"No conversation found for candidate {candidate_id}")
        
        # 1. Speech Recognition
        asr_result = await self.asr_service.transcribe_audio(audio_data)
        transcribed_text = asr_result.get("text", "")
        confidence = asr_result.get("confidence", 0.0)
        
        # 2. Check confidence and handle low confidence
        if confidence < 0.1:
            clarification_text = "Sorry, main aapko samjha nahi. Kya aap dobara bol sakte hain?"
            clarification_audio = await self.tts_service.synthesize_speech(clarification_text)
            
            # Log low confidence turn
            self._log_turn(candidate_id, transcribed_text, confidence, clarification_text, start_time, "low_confidence")
            return clarification_text, clarification_audio, False
        
        # 3. Extract entities using NLU
        current_field = self._get_current_field(state)
        entities = await self.nlu_service.extract_entities(transcribed_text, current_field)
        
        # 4. Update candidate data and state
        response_text, is_completed = await self._update_state_and_generate_response(
            candidate, state, entities, current_field, transcribed_text
        )
        
        # 5. Generate TTS response
        response_audio = await self.tts_service.synthesize_speech(response_text)
        
        # 6. Log the turn
        self._log_turn(candidate_id, transcribed_text, confidence, response_text, start_time, "success")
        
        return response_text, response_audio, is_completed
    
    def _get_current_field(self, state: ConversationState) -> str:
        """Determine what field we're currently collecting and sync current_field."""
        # Clamp step within flow
        if state.current_step < 0:
            state.current_step = 0
        if state.current_step >= len(self.conversation_flow):
            state.current_step = len(self.conversation_flow) - 1
        label = self.conversation_flow[state.current_step]
        state.current_field = label
        return label
    
    async def _update_state_and_generate_response(
        self, 
        candidate: Candidate, 
        state: ConversationState, 
        entities: Dict[str, Any],
        current_field: str,
        user_text: str
    ) -> Tuple[str, bool]:
        """Update candidate data and generate appropriate response."""
        
        # Handle greeting confirmation
        if current_field == "greeting":
            text_lower = user_text.lower()
            positives = ["haan", "yes", "ji", "ready", "tayaar", "taiyar", "tayar"]
            negatives = ["nahi", "no", "nahin"]
            has_positive = any(tok in text_lower for tok in positives)
            has_negative = any(tok in text_lower for tok in negatives)
            if has_positive and not has_negative:
                state.fields_completed.append("greeting")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("pincode"), False
            elif has_positive and has_negative:
                # If mixed signal but contains explicit positive, proceed
                state.fields_completed.append("greeting")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("pincode"), False
            else:
                return "Koi baat nahi! Jab aap tayaar hon tab batayiye।", False
        
        # Handle each field
        if current_field == "pincode":
            if self._extract_and_set_pincode(candidate, entities, user_text):
                state.fields_completed.append("pincode")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("availability_date"), False
            else:
                return "Main aapka area samjha nahi। Kya aap apna pincode ya area naam bata sakte hain?", False
        
        elif current_field == "availability_date":
            if self._extract_and_set_availability(candidate, entities, user_text):
                state.fields_completed.append("availability_date")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("preferred_shift"), False
            else:
                return "Availability clear nahi hui। Kya aap kal se, parso se, ya koi specific date bata sakte hain?", False
        
        elif current_field == "preferred_shift":
            if self._extract_and_set_shift(candidate, entities, user_text):
                state.fields_completed.append("preferred_shift")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("expected_salary"), False
            else:
                return "Shift preference clear nahi। Morning, afternoon, evening, ya night shift - kya prefer karenge?", False
        
        elif current_field == "expected_salary":
            if self._extract_and_set_salary(candidate, entities, user_text):
                state.fields_completed.append("expected_salary")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("languages"), False
            else:
                return "Salary expectation clear nahi। Kitne rupees monthly chahiye? Jaise 15000 ya 20000।", False
        
        elif current_field == "languages":
            if self._extract_and_set_languages(candidate, entities, user_text):
                state.fields_completed.append("languages")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("has_two_wheeler"), False
            else:
                return "Languages clear nahi। Hindi, English, ya koi aur language bata sakte hain?", False
        
        elif current_field == "has_two_wheeler":
            if self._extract_and_set_vehicle(candidate, entities, user_text):
                state.fields_completed.append("has_two_wheeler")
                state.current_step += 1
                state.current_field = self.conversation_flow[state.current_step]
                return self._prompt("total_experience_months"), False
            else:
                return "Vehicle info clear nahi। Kya aapke paas bike ya scooter hai? Haan ya nahi boliye।", False
        
        elif current_field == "total_experience_months":
            if self._extract_and_set_experience(candidate, entities, user_text):
                state.fields_completed.append("total_experience_months")
                state.current_step += 1
                state.current_field = self.conversation_flow[min(state.current_step, len(self.conversation_flow)-1)]
                candidate.conversation_completed = True
                return await self._generate_summary_and_matches(candidate), True
            else:
                return "Experience clear nahi। Kitne mahine ya saal ka experience hai? Ya fresher hain?", False
        
        return "Main samjha nahi। Kya aap dobara bata sakte hain?", False
    
    def _extract_and_set_pincode(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set pincode/locality."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if value:
            import re
            if isinstance(value, str) and re.fullmatch(r"\d{6}", value):
                candidate.pincode = value
                return True
            # treat short textual value as locality
            if isinstance(value, str) and any(ch.isalpha() for ch in value):
                candidate.locality = value.strip().title()
                return True
            # if list provided by model, take first stringy token
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and any(ch.isalpha() for ch in v):
                        candidate.locality = v.strip().title()
                        return True
        # Simple fallback - look for 6-digit number in raw text
        import re
        pincode_match = re.search(r'\b\d{6}\b', text)
        if pincode_match:
            candidate.pincode = pincode_match.group()
            return True
        # Fallbacks for locality from longer utterances
        # Heuristic: take the last 1-2 alphabetic tokens (likely place name)
        tokens = [t for t in re.findall(r"[A-Za-z\u0900-\u097F]+", text)]
        if tokens:
            guess = tokens[-1]
            if len(guess) >= 3:
                candidate.locality = guess.title()
                return True
        # Fallback: if user spoke a short area name, accept as locality
        cleaned = text.strip()
        if any(ch.isalpha() for ch in cleaned) and len(cleaned.split()) <= 6:
            candidate.locality = cleaned.title()
            return True
        return False
    
    def _extract_and_set_availability(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set availability date."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if isinstance(value, str) and value:
            candidate.availability_date = value
            return True
        # Simple fallback
        if any(word in text.lower() for word in ["kal", "tomorrow", "immediately", "abhi", "turant", "aaj", "today"]):
            candidate.availability_date = "immediate" if any(w in text.lower() for w in ["immediately", "abhi", "turant", "now"]) else ("today" if any(w in text.lower() for w in ["aaj", "today"]) else "tomorrow")
            return True
        return False
    
    def _extract_and_set_shift(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set shift preference."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if isinstance(value, str) and value:
            try:
                candidate.preferred_shift = ShiftPreference(value)
                return True
            except ValueError:
                pass
        
        # Fallback mapping
        text_lower = text.lower()
        if any(word in text_lower for word in ["morning", "subah", "day"]):
            candidate.preferred_shift = ShiftPreference.MORNING
            return True
        elif any(word in text_lower for word in ["evening", "sham", "afternoon", "shaam"]):
            candidate.preferred_shift = ShiftPreference.EVENING
            return True
        elif any(word in text_lower for word in ["night", "raat", "rat"]):
            candidate.preferred_shift = ShiftPreference.NIGHT
            return True
        return False
    
    def _extract_and_set_salary(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set expected salary."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if isinstance(value, (int, float)) and value:
            candidate.expected_salary = int(value)
            return True
        
        # Simple fallback - extract numbers
        import re
        numbers = re.findall(r'\d+', text)
        if numbers:
            amount = int(numbers[0])
            # Convert to monthly if seems like annual (> 100k)
            if amount > 100000:
                amount = amount // 12
            candidate.expected_salary = amount
            return True
        return False
    
    def _extract_and_set_languages(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set language skills."""
        value = entities.get("value") if isinstance(entities, dict) else None
        languages = []
        
        # From NLU
        if isinstance(value, list):
            for lang in value:
                if isinstance(lang, str):
                    try:
                        languages.append(LanguageSkill(lang.lower()))
                    except ValueError:
                        pass
        
        # Heuristic fallback
        text_lower = text.lower()
        mapping = [
            ("hindi", LanguageSkill.HINDI),
            ("हिंदी", LanguageSkill.HINDI),
            ("english", LanguageSkill.ENGLISH),
            ("angrezi", LanguageSkill.ENGLISH),
            ("gujarati", LanguageSkill.GUJARATI),
            ("गुजराती", LanguageSkill.GUJARATI),
            ("marathi", LanguageSkill.MARATHI),
            ("मराठी", LanguageSkill.MARATHI),
        ]
        for key, enum_val in mapping:
            if key in text_lower and enum_val not in languages:
                languages.append(enum_val)
        
        if languages:
            candidate.languages = languages
            return True
        return False
    
    def _extract_and_set_vehicle(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set vehicle ownership."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if isinstance(value, bool):
            candidate.has_two_wheeler = value
            return True
        
        text_lower = text.lower()
        if any(word in text_lower for word in ["haan", "yes", "hai", "bike", "scooter"]):
            candidate.has_two_wheeler = True
            return True
        elif any(word in text_lower for word in ["nahi", "no", "nahin"]):
            candidate.has_two_wheeler = False
            return True
        return False
    
    def _extract_and_set_experience(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set total experience."""
        value = entities.get("value") if isinstance(entities, dict) else None
        if isinstance(value, (int, float)):
            candidate.total_experience_months = int(value)
            return True
        
        # Simple fallback
        import re
        text_lower = text.lower()
        if any(word in text_lower for word in ["fresher", "नया", "naya", "koi nahi", "no experience"]):
            candidate.total_experience_months = 0
            return True
        
        # Extract years/months
        years_match = re.search(r'(\d+)\s*(?:years?|saal)', text_lower)
        months_match = re.search(r'(\d+)\s*(?:months?|mahine|mahina)', text_lower)
        
        total_months = 0
        if years_match:
            total_months += int(years_match.group(1)) * 12
        if months_match:
            total_months += int(months_match.group(1))
        
        if total_months > 0:
            candidate.total_experience_months = total_months
            return True
        return False
    
    async def _generate_summary_and_matches(self, candidate: Candidate) -> str:
        """Generate summary and find job matches."""
        # Get job matches
        matching_result = await self.job_matching_service.find_job_matches(candidate)
        
        # Personalized Hinglish summary (LLM-backed if available)
        personalized = await self.job_matching_service.generate_personalized_summary(candidate, matching_result)
        return personalized
    
    def _log_turn(self, candidate_id: str, user_text: str, confidence: float, 
                  bot_response: str, start_time: float, status: str):
        """Log conversation turn for analytics."""
        total_latency_ms = int((time.time() - start_time) * 1000)
        turn = VoiceTurn(
            turn_id=str(uuid.uuid4()),
            candidate_id=candidate_id,
            asr_text=user_text,
            asr_confidence=confidence,
            extracted_entities={},
            chosen_prompt=self._get_current_field(self.conversation_states[candidate_id]),
            tts_text=bot_response,
            tts_char_count=len(bot_response),
            asr_latency_ms=0.0,
            nlu_latency_ms=0.0,
            tts_latency_ms=0.0,
            total_latency_ms=total_latency_ms,
        )
        
        if candidate_id not in self.conversation_turns:
            self.conversation_turns[candidate_id] = []
        self.conversation_turns[candidate_id].append(turn)
        
        # Increment candidate turn count
        self.candidates[candidate_id].turn_count += 1
        
        # Log for metrics
        logger.bind(conversation=True).info({
            "event": "conversation_turn",
            "candidate_id": candidate_id,
            "turn_number": len(self.conversation_turns[candidate_id]),
            "user_text": user_text,
            "confidence": confidence,
            "response_length": len(bot_response),
            "processing_time_ms": total_latency_ms,
            "status": status
        })
        
    def _prompt(self, field_key: str) -> str:
        """Resolve the correct Hinglish prompt for a given field key."""
        mapped = self.prompt_key_map.get(field_key, field_key)
        return self.prompts.get(mapped, self.prompts.get("error_generic", "Sorry, samajh nahi aaya."))

    def get_conversation_metrics(self, candidate_id: str) -> Optional[Dict[str, Any]]:
        """Aggregate metrics for a conversation."""
        turns = self.conversation_turns.get(candidate_id, [])
        if not turns:
            return None
        latencies = [t.total_latency_ms for t in turns]
        confidences = [t.asr_confidence for t in turns]
        
        def percentile(values, p):
            if not values:
                return 0.0
            values = sorted(values)
            k = int(round((p/100) * (len(values)-1)))
            return values[k]
        
        state = self.conversation_states.get(candidate_id)
        candidate = self.candidates.get(candidate_id)
        completion_rate = state.completion_rate if state else 0.0
        current_field = state.current_field if state else None
        
        return {
            "total_turns": len(turns),
            "avg_latency_ms": sum(latencies)/len(latencies),
            "avg_confidence": sum(confidences)/len(confidences),
            "p50_latency_ms": percentile(latencies, 50),
            "p95_latency_ms": percentile(latencies, 95),
            "completion_rate": completion_rate,
            "current_field": current_field,
            "candidate_profile": candidate.dict() if candidate else None,
        } 