import asyncio
import uuid
import time
from typing import Dict, Optional, Tuple, Any, List
from datetime import datetime

from app.models.candidate import Candidate, ConversationState, VoiceTurn, ShiftPreference, LanguageSkill
from app.services.speech_recognition import ASRService

from app.services.text_to_speech import TTSService
from app.services.nlu import NLUService
from app.services.job_matching import JobMatchingService
from app.core.config import settings
from app.core.logger import logger

import openai


class ConversationOrchestrator:
    """Orchestrates the voice conversation flow for candidate screening."""
    
    def __init__(self):
        # Initialize services individually to catch specific errors
        try:
            # Use Google Speech API for real confidence scores
            self.asr_service = ASRService()
            logger.info("Google Speech API ASR service initialized successfully")
        except Exception as e:
            logger.error(f"Google Speech API ASR service initialization failed: {e}")
            self.asr_service = None
        
        try:
            self.tts_service = TTSService()
            logger.info("TTS service initialized successfully")
        except Exception as e:
            logger.error(f"TTS service initialization failed: {e}")
            import traceback
            traceback.print_exc()
            self.tts_service = None
        
        try:
            self.nlu_service = NLUService()
            logger.info("NLU service initialized successfully")
        except Exception as e:
            logger.error(f"NLU service initialization failed: {e}")
            self.nlu_service = None
        
        self.job_matching_service = JobMatchingService()
        
        # In-memory storage for demo (use database in production)
        self.candidates: Dict[str, Candidate] = {}
        self.conversation_states: Dict[str, ConversationState] = {}
        self.conversation_turns: Dict[str, list] = {}
        
        # Background TTS pending audio store: candidate_id -> {turn_id: audio_bytes}
        self.pending_audio: Dict[str, Dict[str, bytes]] = {}
        
        # Conversation flow definition
        self.conversation_flow = [
            "greeting",
            "pincode",
            "expected_salary",
            "has_two_wheeler",
            "languages",
            "availability_date",
            "preferred_shift",
            "total_experience_months",
            "confirmation",
            "summary"
        ]
        
        # Define Hinglish prompts directly in conversation service
        self.prompts = {
            "greeting": "Namaste! Main aapka voice assistant hun job interview ke liye। Yeh call record ho rahi hai। Kya aap tayaar hain?",
            "pincode": "Aap kahan rehte hain? Apna pincode ya area batayiye।",
            "pincode_confirm": "Aapne {value} kaha, sahi hai na?",
            "pincode_retry": "Pincode samajh nahi aaya। 6 digit number boliye jaise 110001।",
            "availability": "Aap kab se kaam shuru kar sakte hain? Aaj, kal ya koi aur din?",
            "availability_confirm": "Toh aap {value} se start kar sakte hain, correct?",
            "availability_retry": "Date samajh nahi aayi। Aaj, kal, parso - aise boliye।",
            "shift": "Aap kaunse time pe kaam karna chahte hain? Morning, evening ya night?",
            "shift_confirm": "Aap {value} shift prefer karte hain, right?",
            "shift_retry": "Shift samajh nahi aayi। Morning, afternoon, evening ya night - koi ek choose kariye।",
            "salary": "Aapko kitni salary chahiye har mahine? Rupees mein batayiye।",
            "salary_confirm": "Aapki expected salary {value} rupees per month hai, sahi?",
            "salary_retry": "Salary amount clear nahi hai। Number mein boliye jaise 15 hazaar।",
            "languages": "Aap kaunsi languages bol sakte hain? Hindi, English ya koi aur?",
            "languages_confirm": "Aap {value} bol sakte hain, confirm hai?",
            "languages_retry": "Languages samajh nahi aayi। Hindi, English - aise batayiye।",
            "two_wheeler": "Kya aapke paas bike ya scooter hai?",
            "two_wheeler_confirm": "Aapke paas two wheeler {value} hai, right?",
            "two_wheeler_retry": "Haan ya nahi mein jawab dijiye। Bike hai ya nahi?",
            "experience": "Aapko kitna kaam ka experience hai? Kitne saal ya mahine?",
            "experience_confirm": "Aapka total experience {value} hai, correct?",
            "experience_retry": "Experience time samajh nahi aaya। Months ya years mein batayiye।",
            "summary": "Perfect! Main aapki details note kar li। Aapko {locality} area mein, {salary} salary range mein best jobs bhejunga। Thank you!",
            "error_generic": "Sorry, samajh nahi aaya। Dobara boliye।",
            "error_noise": "Background noise zyada hai। Shant jagah se baat kariye।",
            "error_timeout": "Aapki awaaz nahi aa rahi। Phone check kariye।",
            "goodbye": "Aapka interview complete ho gaya। Job matches SMS mein aayenge। Dhanyawad!"
        }
        
        # Mapping from state field to prompt key in prompts
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
    
    async def synthesize_and_store_audio(self, candidate_id: str, turn_id: str, text: str) -> None:
        """Generate TTS in background and store until client fetches it."""
        try:
            if not self.tts_service:
                return
            audio_bytes = await self.tts_service.synthesize_speech(text)
            if candidate_id not in self.pending_audio:
                self.pending_audio[candidate_id] = {}
            self.pending_audio[candidate_id][turn_id] = audio_bytes
        except Exception as e:
            logger.error(f"Background TTS failed for {candidate_id}/{turn_id}: {e}")

    def pop_pending_audio(self, candidate_id: str, turn_id: str) -> Optional[bytes]:
        """Return and remove pending audio if present."""
        try:
            if candidate_id in self.pending_audio and turn_id in self.pending_audio[candidate_id]:
                audio = self.pending_audio[candidate_id].pop(turn_id)
                # Cleanup candidate bucket if empty
                if not self.pending_audio[candidate_id]:
                    del self.pending_audio[candidate_id]
                return audio
        except Exception:
            pass
        return None

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
        
        # Log conversation start early
        logger.bind(conversation=True).info({
            "event": "conversation_started",
            "candidate_id": candidate_id,
            "state_stored": True
        })
        
        try:
            # Generate initial greeting
            greeting_text = self.prompts["greeting"]
            audio_data = await self.tts_service.synthesize_speech(greeting_text)
        except Exception as e:
            # If TTS fails, still return the conversation but with empty audio
            logger.warning(f"TTS failed during conversation start: {e}")
            audio_data = b""  # Empty audio data
        
        return candidate_id, audio_data
    
    async def process_turn_text_only(self, candidate_id: str, audio_data: bytes) -> Tuple[str, bool, str, float, Dict[str, Any], str]:
        """Process a turn up to response_text (no TTS), return turn_id for later audio fetch."""
        if self.asr_service is None or self.nlu_service is None:
            raise Exception("Voice services not available - Google Cloud APIs need to be enabled")
        start_time = time.time()
        state = self.conversation_states.get(candidate_id)
        candidate = self.candidates.get(candidate_id)
        if not state or not candidate:
            raise ValueError(f"No conversation found for candidate {candidate_id}")
        asr_result = await self.asr_service.transcribe_audio(audio_data)
        transcribed_text = asr_result.get("text", "")
        confidence = asr_result.get("confidence", 0.0)
        raw_confidence_data = asr_result.get("raw_confidence_data", {})
        
        # Log detailed ASR information
        logger.info(f"ASR heard: '{transcribed_text}' (conf {confidence:.2f})")
        logger.info(f"ASR provider: {raw_confidence_data.get('asr_provider', 'unknown')}")
        logger.info(f"Confidence source: {raw_confidence_data.get('confidence_source', 'unknown')}")
        history_pairs = []
        for t in self.conversation_turns.get(candidate_id, [])[-8:]:
            if t.asr_text:
                history_pairs.append(("User", t.asr_text))
            if t.tts_text:
                history_pairs.append(("Bot", t.tts_text))

        # If awaiting a confirmation, classify yes/no via LLM and branch
        if state.needs_confirmation and state.pending_confirmation_slot:
            # First, allow user to provide a new value directly instead of yes/no
            slot = state.pending_confirmation_slot
            try:
                entities = await self.nlu_service.extract_entities(transcribed_text, slot)
                if self._try_update_slot(candidate, state, slot, entities, transcribed_text):
                    # Accept new value and clear confirmation
                    state.needs_confirmation = False
                    state.pending_confirmation_value = None
                    state.pending_confirmation_slot = None
                    response_text, target_slot, is_completed = await self._generate_intelligent_response(
                        transcribed_text, confidence, candidate, state, history_pairs
                    )
                else:
                    # Fallback to yes/no classification
                    decision = await self._classify_affirmation_via_llm(
                        transcribed_text,
                        history_pairs,
                        slot,
                        state.pending_confirmation_value
                    )
                    if decision == "yes":
                        self._apply_confirmed_value(candidate, state, slot, state.pending_confirmation_value)
                        state.needs_confirmation = False
                        state.pending_confirmation_value = None
                        state.pending_confirmation_slot = None
                        response_text, target_slot, is_completed = await self._generate_intelligent_response(
                            transcribed_text, confidence, candidate, state, history_pairs
                        )
                    else:
                        prompt_key = self.prompt_key_map.get(slot)
                        if prompt_key and prompt_key in self.prompts:
                            response_text = self.prompts[prompt_key]
                        else:
                            response_text = f"Kya aap bata sakte hain aapka {self._get_slot_description(slot)}?"
                        state.needs_confirmation = False
                        state.pending_confirmation_value = None
                        state.pending_confirmation_slot = None
                        target_slot = slot
                        is_completed = False
            except Exception as e:
                logger.error(f"Confirmation handling error: {e}")
                # Safe fallback to re-ask
                prompt_key = self.prompt_key_map.get(slot)
                response_text = self.prompts.get(prompt_key, f"Kya aap bata sakte hain aapka {self._get_slot_description(slot)}?")
                state.needs_confirmation = False
                state.pending_confirmation_value = None
                state.pending_confirmation_slot = None
                target_slot = slot
                is_completed = False
        else:
            response_text, target_slot, is_completed = await self._generate_intelligent_response(
                transcribed_text, confidence, candidate, state, history_pairs
            )
        # Generate a turn_id now and log the turn without audio (tts latency 0)
        turn_id = str(uuid.uuid4())
        status = "completed" if is_completed else ("intelligent" if target_slot else "contextual")
        self._log_turn(candidate_id, transcribed_text, confidence, response_text, start_time, status, turn_id=turn_id)
        return response_text, is_completed, transcribed_text, confidence, raw_confidence_data, turn_id

    async def process_turn(self, candidate_id: str, audio_data: bytes) -> Tuple[str, bytes, bool, str, float]:
        """Process a voice turn from the candidate.
        
        Args:
            candidate_id: The candidate's unique ID
            audio_data: Raw audio bytes from the candidate
            
        Returns:
            Tuple[str, bytes, bool, str, float]: (response_text, response_audio, is_completed, asr_text, asr_confidence)
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
        
        # Add ASR debug line into history and logs
        logger.info(f"ASR heard: '{transcribed_text}' (conf {confidence:.2f})")
        
        # 2. Build conversation history
        history_pairs = []
        for t in self.conversation_turns.get(candidate_id, [])[-8:]:
            if t.asr_text:
                history_pairs.append(("User", t.asr_text))
            if t.tts_text:
                history_pairs.append(("Bot", t.tts_text))
        
        # 3. LLM-driven response generation
        response_text, target_slot, is_completed = await self._generate_intelligent_response(
            transcribed_text, confidence, candidate, state, history_pairs
        )
        
        # 4. Generate TTS response
        response_audio = await self.tts_service.synthesize_speech(response_text)
        
        # 5. Log the turn
        status = "completed" if is_completed else ("intelligent" if target_slot else "contextual")
        self._log_turn(candidate_id, transcribed_text, confidence, response_text, start_time, status)
        
        return response_text, response_audio, is_completed, transcribed_text, confidence

    async def _generate_intelligent_response(
        self, 
        user_text: str, 
        confidence: float,
        candidate: Candidate, 
        state: ConversationState,
        history: List[Tuple[str, str]]
    ) -> Tuple[str, Optional[str], bool]:
        """Generate intelligent response using LLM with tool-based conversation management.
        
        Returns:
            Tuple[str, Optional[str], bool]: (response_text, target_slot, is_completed)
        """
        # Get missing slots
        required_slots = ["pincode", "availability_date", "preferred_shift", "expected_salary", 
                         "languages", "has_two_wheeler", "total_experience_months", "confirmation"]
        completed_slots = set(state.fields_completed)
        missing_slots = [s for s in required_slots if s not in completed_slots]
        
        # Special handling for confirmation step
        if "confirmation" in missing_slots and len(missing_slots) == 1:
            return await self._handle_confirmation_step(candidate, state, user_text, confidence)
        
        # Get next slot to collect (skip confirmation for regular flow)
        extraction_slots = [s for s in missing_slots if s != "confirmation"]
        next_slot = self._choose_next_slot(extraction_slots, user_text, False)

        # Sync current_field and reset retry counter when moving to a new slot
        if state.current_field != next_slot:
            state.current_field = next_slot
            state.retry_count = 0
        
        # If user provided input, try to extract information (confidence only applies to entity extraction)
        extracted_something = False
        if user_text.strip() and confidence >= settings.confidence_threshold:
            # Try to extract for the next expected slot first
            entities = await self.nlu_service.extract_entities(user_text, next_slot)
            if self._try_update_slot(candidate, state, next_slot, entities, user_text):
                extracted_something = True
                state.retry_count = 0
                logger.info(f"Successfully extracted {next_slot} from user input: {user_text}")
            
            # If that didn't work, try other missing slots
            if not extracted_something:
                for slot in extraction_slots:
                    if slot != next_slot:  # Skip the one we already tried
                        entities = await self.nlu_service.extract_entities(user_text, slot)
                        if self._try_update_slot(candidate, state, slot, entities, user_text):
                            extracted_something = True
                            state.current_field = slot
                            state.retry_count = 0
                            logger.info(f"Successfully extracted {slot} from user input: {user_text}")
                            break
        
        # If rule-based extraction struggled for this slot more than twice, escalate to LLM
        if not extracted_something and user_text.strip() and state.retry_count >= 2 and settings.openai_api_key:
            try:
                entities = await self.nlu_service._extract_with_openai(user_text, next_slot)
                if self._try_update_slot(candidate, state, next_slot, entities, user_text):
                    extracted_something = True
                    state.retry_count = 0
                    logger.info(f"LLM extraction succeeded for {next_slot}: {entities}")
            except Exception as e:
                logger.error(f"LLM extraction failed for {next_slot}: {e}")
        
        # If ASR confidence is low, propose extracted entity and ask for confirmation instead of updating directly
        if user_text.strip() and not extracted_something and confidence < settings.confidence_threshold:
            try:
                entities = await self.nlu_service.extract_entities(user_text, next_slot)
                value = entities.get("value") if isinstance(entities, dict) else None
                if value is not None and value != "":
                    # Build a readable value for confirmation
                    if isinstance(value, list):
                        display_value = ", ".join([str(v) for v in value if v is not None])
                    else:
                        display_value = str(value)
                    state.needs_confirmation = True
                    state.pending_confirmation_value = display_value
                    state.pending_confirmation_slot = next_slot
                    confirm_text = self._build_low_conf_confirmation(next_slot, display_value)
                    return confirm_text, next_slot, False
            except Exception as e:
                logger.error(f"Low-confidence provisional extraction failed: {e}")
        
        # Refresh missing slots after potential extraction
        completed_slots = set(state.fields_completed)
        missing_slots = [s for s in required_slots if s not in completed_slots]
        
        logger.info(f"Conversation flow: completed_slots={completed_slots}, missing_slots={missing_slots}")
        
        # Check if we're done with data collection (only confirmation left)
        if len(missing_slots) == 1 and "confirmation" in missing_slots:
            logger.info("Triggering confirmation step - only confirmation is missing")
            return await self._handle_confirmation_step(candidate, state, "", confidence)
        
        # Recalculate missing slots after potential entity extraction
        if extracted_something:
            completed_slots = set(state.fields_completed)
            missing_slots = [s for s in required_slots if s not in completed_slots]
            logger.info(f"Recalculated missing slots after extraction: {missing_slots}")
        
        # Check if completely done
        if not missing_slots:
            logger.info("All slots completed - generating summary and matches")
            summary_text = await self._generate_summary_and_matches(candidate)
            candidate.conversation_completed = True
            return summary_text, None, True
        
        # Get next slot to ask for
        extraction_slots = [s for s in missing_slots if s != "confirmation"]
        next_slot = self._choose_next_slot(extraction_slots, user_text, extracted_something)
        
        # If moving to a new slot, reset retry counter
        if state.current_field != next_slot:
            state.current_field = next_slot
            state.retry_count = 0
        
        # If we extracted something, use default prompt for next slot
        if extracted_something:
            prompt_key = self.prompt_key_map.get(next_slot)
            if prompt_key and prompt_key in self.prompts:
                response_text = self.prompts[prompt_key]
                return response_text, next_slot, False
        
        # Confirmation-style rephrase when we heard something but couldn't extract entities
        if user_text.strip() and not extracted_something:
            state.retry_count += 1
            # Clean the heard text for readable confirmation (keep short location-like tokens)
            cleaned = self._normalize_confirmation_value(next_slot, user_text)
            state.needs_confirmation = True
            state.pending_confirmation_value = cleaned
            state.pending_confirmation_slot = next_slot
            confirm_text = self._build_low_conf_confirmation(next_slot, cleaned)
            return confirm_text, next_slot, False
        
        # Default: ask for next slot using standard or retry prompt based on attempts
        prompt_key = self.prompt_key_map.get(next_slot)
        if state.retry_count > 0:
            retry_key = f"{next_slot}_retry"
            if retry_key in self.prompts:
                state.retry_count += 1
                response_text = self.prompts[retry_key]
                return response_text, next_slot, False
        if prompt_key and prompt_key in self.prompts:
            # First time asking this slot
            state.retry_count += 1
            response_text = self.prompts[prompt_key]
        else:
            slot_desc = self._get_slot_description(next_slot)
            state.retry_count += 1
            response_text = f"Kya aap bata sakte hain aapka {slot_desc}?"
        
        return response_text, next_slot, False

    async def _generate_contextual_response(self, user_text: str, next_slot: str, candidate: Candidate, history: List[Tuple[str, str]]) -> Tuple[str, Optional[str], bool]:
        """Generate contextual response when user input is unexpected or unclear."""
        history_text = "\n".join([f"{speaker}: {text}" for speaker, text in history[-4:]])
        current_data = self._get_candidate_summary(candidate)
        
        system_prompt = (
            "You are a Hinglish voice assistant conducting a job interview. "
            "The user said something unexpected or unclear. Acknowledge their input politely "
            "and then ask for the specific information needed. Keep response under 50 words."
        )
        
        slot_desc = self._get_slot_description(next_slot)
        user_prompt = (
            f"CONVERSATION HISTORY:\n{history_text}\n\n"
            f"CURRENT DATA:\n{current_data}\n\n"
            f"USER SAID: '{user_text}'\n\n"
            f"NEXT INFORMATION NEEDED: {slot_desc}\n\n"
            f"Generate a natural Hinglish response that acknowledges their input "
            f"and asks for {slot_desc}."
        )
        
        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=80,
                temperature=0.3
            )
            
            return response.choices[0].message.content.strip(), next_slot, False
            
        except Exception as e:
            logger.error(f"Contextual response generation failed: {e}")
            return f"Samjha. Kya aap bata sakte hain aapka {slot_desc}?", next_slot, False

    async def _handle_confirmation_step(self, candidate: Candidate, state: ConversationState, user_text: str, confidence: float) -> Tuple[str, Optional[str], bool]:
        """Enhanced confirmation step using comprehensive entity validation system."""
        logger.info(f"Entering confirmation step with user_text='{user_text}', fields_completed={state.fields_completed}")
        
        try:
            # Check for missing or invalid data first
            missing_fields = self._check_missing_required_fields(candidate)
            
            if missing_fields:
                logger.info(f"Missing fields detected in confirmation: {missing_fields}")
                # Re-ask for the first missing field
                first_missing = missing_fields[0]
                if first_missing in state.fields_completed:
                    state.fields_completed.remove(first_missing)
                
                prompt_key = self.prompt_key_map.get(first_missing)
                if prompt_key and prompt_key in self.prompts:
                    response_text = f"Sorry, {self._get_slot_description(first_missing)} ki information complete nahi hai. {self.prompts[prompt_key]}"
                else:
                    slot_desc = self._get_slot_description(first_missing)
                    response_text = f"Sorry, {slot_desc} ki information complete nahi hai. Kya aap bata sakte hain aapka {slot_desc}?"
                
                return response_text, first_missing, False
            
            # If no user input (first time in confirmation), show summary
            if not user_text.strip():
                logger.info("Showing confirmation summary")
                # Try to resolve pincode from locality if missing
                await self._resolve_missing_pincode(candidate)
                
                # Use original confirmation method as primary
                try:
                    logger.info(f"Trying original confirmation method for candidate: pincode={candidate.pincode}, locality={candidate.locality}, salary={candidate.expected_salary}")
                    confirmation_summary = await self._generate_confirmation_summary(candidate)
                    logger.info(f"Original confirmation summary generated successfully: {confirmation_summary[:100]}...")
                    state.in_final_confirmation = True
                    return confirmation_summary, "confirmation", False
                except Exception as e:
                    import traceback
                    logger.error(f"Original confirmation method failed: {e}")
                    logger.error(f"Original confirmation traceback: {traceback.format_exc()}")
                    # Fallback to new comprehensive entity confirmation summary
                    candidate_data = self._extract_candidate_data_dict(candidate)
                    logger.info(f"Using new confirmation method with candidate data: {candidate_data}")
                    confirmation_summary = await self.nlu_service.generate_entity_confirmation_summary(candidate_data)
                    logger.info(f"New confirmation summary generated: {confirmation_summary[:100]}...")
                    state.in_final_confirmation = True
                    return confirmation_summary, "confirmation", False
        
            # Process user's confirmation response
            # Use new LLM method for better change detection, with fallback to original
            try:
                candidate_data = self._extract_candidate_data_dict(candidate)
                logger.info(f"Processing user confirmation response: '{user_text}' using new method")
                confirmation_result = await self.nlu_service.process_entity_confirmation_response(
                    user_text, candidate_data, confidence
                )
            except Exception as response_error:
                logger.error(f"New confirmation response processing failed, falling back to original: {response_error}")
                # Fallback to original simple affirmation check
                if self._is_affirmative(user_text):
                    confirmation_result = {
                        "action": "accept",
                        "changes": {},
                        "confirmation_text": "Perfect! Saari details confirm ho gayi.",
                        "needs_reconfirmation": False
                    }
                else:
                    confirmation_result = {
                        "action": "clarify", 
                        "changes": {},
                        "confirmation_text": "Samjha nahi. Kya saari information sahi hai? Haan ya nahi mein jawab dijiye.",
                        "needs_reconfirmation": True
                    }
            
            action = confirmation_result.get("action", "clarify")
            changes = confirmation_result.get("changes", {})
            response_text = confirmation_result.get("confirmation_text", "")
            needs_reconfirmation = confirmation_result.get("needs_reconfirmation", True)
            
            logger.info(f"Confirmation result: action={action}, changes={changes}")
            
            if action == "accept":
                # User accepted all entities - complete conversation
                logger.info("User accepted all entities - completing conversation")
                if "confirmation" not in state.fields_completed:
                    state.fields_completed.append("confirmation")
                summary_text = await self._generate_summary_and_matches(candidate)
                candidate.conversation_completed = True
                state.in_final_confirmation = False
                return summary_text, None, True
                
            elif action == "modify" and changes:
                # Apply detected changes to candidate
                logger.info(f"Applying changes: {changes}")
                updated_entities = await self.nlu_service.apply_entity_changes(
                    candidate_data, changes, user_text
                )
                
                # Update candidate object with new values
                self._update_candidate_from_entities(candidate, updated_entities)
                
                if needs_reconfirmation:
                    # Generate new confirmation summary with updated values
                    candidate_data = self._extract_candidate_data_dict(candidate)
                    new_summary = await self.nlu_service.generate_entity_confirmation_summary(candidate_data)
                    return new_summary, "confirmation", False
                else:
                    # Changes applied and confirmed - complete conversation
                    if "confirmation" not in state.fields_completed:
                        state.fields_completed.append("confirmation")
                    summary_text = await self._generate_summary_and_matches(candidate)
                    candidate.conversation_completed = True
                    state.in_final_confirmation = False
                    return summary_text, None, True
                    
            else:  # clarify or unknown action
                # Ask for clarification
                return response_text, "confirmation", False
                
        except Exception as e:
            logger.error(f"Error in confirmation step: {e}")
            import traceback
            logger.error(f"Confirmation step traceback: {traceback.format_exc()}")
            # Fallback to original confirmation method first
            try:
                logger.info("Falling back to original confirmation method in error handler")
                summary = await self._generate_confirmation_summary(candidate)
                logger.info(f"Fallback original confirmation worked: {summary[:100]}...")
                return summary, "confirmation", False
            except Exception as fallback_e:
                import traceback
                logger.error(f"Original confirmation fallback also failed: {fallback_e}")
                logger.error(f"Fallback traceback: {traceback.format_exc()}")
                # Try new confirmation method as secondary fallback
                try:
                    candidate_data = self._extract_candidate_data_dict(candidate)
                    confirmation_summary = await self.nlu_service.generate_entity_confirmation_summary(candidate_data)
                    logger.info(f"Secondary fallback new confirmation worked: {confirmation_summary[:100]}...")
                    return confirmation_summary, "confirmation", False
                except Exception as final_fallback_e:
                    logger.error(f"Even new confirmation method failed: {final_fallback_e}")
                    logger.error(f"Final fallback traceback: {traceback.format_exc()}")
                    # Last resort fallback
                    logger.warning("Using last resort generic confirmation message")
                    return "Kya aapki saari details sahi hain? Haan ya nahi mein jawab dijiye.", "confirmation", False
    
    def _check_missing_required_fields(self, candidate: Candidate) -> List[str]:
        """Check for missing required fields in candidate profile."""
        missing_fields = []
        
        # Location (pincode or locality)
        if not candidate.pincode and not candidate.locality:
            missing_fields.append("pincode")
        
        # Other required fields
        if not candidate.availability_date:
            missing_fields.append("availability_date")
        if not candidate.preferred_shift:
            missing_fields.append("preferred_shift")
        if not candidate.expected_salary:
            missing_fields.append("expected_salary")
        if not candidate.languages and not candidate.other_languages:
            missing_fields.append("languages")
        if candidate.has_two_wheeler is None:
            missing_fields.append("has_two_wheeler")
        if candidate.total_experience_months is None:
            missing_fields.append("total_experience_months")
            
        return missing_fields
    
    def _extract_candidate_data_dict(self, candidate: Candidate) -> Dict[str, Any]:
        """Extract candidate data into a dictionary for entity processing."""
        return {
            "pincode": candidate.pincode,
            "locality": candidate.locality,
            "expected_salary": candidate.expected_salary,
            "total_experience_months": candidate.total_experience_months,
            "languages": candidate.languages,
            "other_languages": candidate.other_languages,
            "availability_date": candidate.availability_date,
            "preferred_shift": candidate.preferred_shift,
            "has_two_wheeler": candidate.has_two_wheeler,
        }
    
    def _update_candidate_from_entities(self, candidate: Candidate, entities: Dict[str, Any]):
        """Update candidate object from entities dictionary."""
        if "pincode" in entities and entities["pincode"]:
            candidate.pincode = entities["pincode"]
            candidate.locality = None  # Clear locality if pincode is set
        if "locality" in entities and entities["locality"]:
            candidate.locality = entities["locality"]
            if not candidate.pincode:  # Only clear pincode if we don't have one
                candidate.pincode = None
                
        if "expected_salary" in entities and entities["expected_salary"]:
            candidate.expected_salary = entities["expected_salary"]
            
        if "total_experience_months" in entities and entities["total_experience_months"] is not None:
            candidate.total_experience_months = entities["total_experience_months"]
            
        if "languages" in entities and entities["languages"]:
            # Handle LanguageSkill enum conversion
            from app.models.candidate import LanguageSkill
            if isinstance(entities["languages"], list):
                lang_enums = []
                for lang in entities["languages"]:
                    if isinstance(lang, str):
                        try:
                            lang_enums.append(LanguageSkill(lang.lower()))
                        except ValueError:
                            # Add to other_languages if not a valid enum
                            if not candidate.other_languages:
                                candidate.other_languages = []
                            if lang.title() not in candidate.other_languages:
                                candidate.other_languages.append(lang.title())
                    elif isinstance(lang, LanguageSkill):
                        lang_enums.append(lang)
                candidate.languages = lang_enums
                
        if "other_languages" in entities and entities["other_languages"]:
            candidate.other_languages = entities["other_languages"]
            
        if "availability_date" in entities and entities["availability_date"]:
            candidate.availability_date = entities["availability_date"]
            
        if "preferred_shift" in entities and entities["preferred_shift"]:
            from app.models.candidate import ShiftPreference
            if isinstance(entities["preferred_shift"], str):
                try:
                    candidate.preferred_shift = ShiftPreference(entities["preferred_shift"].lower())
                except ValueError:
                    # Handle invalid shift values gracefully
                    candidate.preferred_shift = ShiftPreference.FLEXIBLE
            elif isinstance(entities["preferred_shift"], ShiftPreference):
                candidate.preferred_shift = entities["preferred_shift"]
                
        if "has_two_wheeler" in entities and entities["has_two_wheeler"] is not None:
            candidate.has_two_wheeler = entities["has_two_wheeler"]
    
    async def _resolve_missing_pincode(self, candidate: Candidate) -> bool:
        """Automatically resolve pincode from locality if pincode is missing but locality exists."""
        # Only resolve if we have locality but no pincode
        if candidate.locality and not candidate.pincode:
            try:
                resolved_pincode = await self.nlu_service.resolve_pincode_from_locality(candidate.locality)
                if resolved_pincode:
                    candidate.pincode = resolved_pincode
                    logger.info(f"Auto-resolved pincode '{resolved_pincode}' from locality '{candidate.locality}'")
                    return True
            except Exception as e:
                logger.error(f"Failed to auto-resolve pincode for locality '{candidate.locality}': {e}")
        
        return False

    async def _generate_confirmation_summary(self, candidate: Candidate) -> str:
        """Generate a summary of collected information for confirmation."""
        # Build detailed summary of collected information
        summary_parts = ["Dhanyawad! Main aapki details repeat karta hun:"]
        
        # Helper function to check if a value is valid (not an error message)
        def is_valid_value(value):
            if not value:
                return False
            if isinstance(value, str):
                # Check for common error phrases that indicate malformed data
                error_phrases = ["sorry", "could you provide", "does not specify", "not clear", "unclear"]
                return not any(phrase in value.lower() for phrase in error_phrases)
            return True
        
        if candidate.pincode:
            summary_parts.append(f"Pincode: {candidate.pincode},")
        elif candidate.locality:
            summary_parts.append(f"Area: {candidate.locality},")
        
        if candidate.availability_date and is_valid_value(candidate.availability_date):
            summary_parts.append(f"Availability: {candidate.availability_date},")
        
        if candidate.preferred_shift:
            try:
                if hasattr(candidate.preferred_shift, 'value'):
                    shift_display = candidate.preferred_shift.value.replace('_', ' ').title()
                else:
                    shift_display = str(candidate.preferred_shift).replace('_', ' ').title()
                summary_parts.append(f"aapka Preferred shift: {shift_display} hai,")
            except Exception as shift_error:
                logger.error(f"Error formatting shift in confirmation: {shift_error}")
                summary_parts.append(f"aapka Preferred shift: {str(candidate.preferred_shift)} hai,")
        
        if candidate.expected_salary:
            summary_parts.append(f"Expected salary: ₹{candidate.expected_salary:,} per month,")
        
        if candidate.languages or candidate.other_languages:
            lang_list = []
            try:
                if candidate.languages:
                    for l in candidate.languages:
                        if hasattr(l, 'value'):
                            lang_list.append(l.value.title())
                        else:
                            lang_list.append(str(l).title())
                if candidate.other_languages:
                    lang_list.extend([f"{l}" for l in candidate.other_languages])
                summary_parts.append(f"aap {', '.join(lang_list)} bol sakte hain,")
            except Exception as lang_error:
                logger.error(f"Error formatting languages in confirmation: {lang_error}")
                summary_parts.append(f"aap languages bol sakte hain,")
        
        if candidate.has_two_wheeler is not None:
            two_wheeler_status = "Hai" if candidate.has_two_wheeler else "Nahi hai"
            summary_parts.append(f"aapke paas Two wheeler: {two_wheeler_status}. ")
        
        if candidate.total_experience_months is not None:
            if candidate.total_experience_months == 0:
                summary_parts.append("Experience: Fresher (0 months)")
            else:
                years = candidate.total_experience_months // 12
                months = candidate.total_experience_months % 12
                if years > 0:
                    if months > 0:
                        summary_parts.append(f"Experience: {years} year{'s' if years > 1 else ''} {months} month{'s' if months > 1 else ''}")
                    else:
                        summary_parts.append(f"Experience: {years} year{'s' if years > 1 else ''}")
                else:
                    summary_parts.append(f"Experience: {months} month{'s' if months > 1 else ''}")
        
        # Join all parts and add confirmation question
        summary_text = "\n".join(summary_parts)
        summary_text += "\n\nKya yeh saari information jo humne collect ki hai, sahi hai? Agar kuch galat hai toh please bataiye."
        
        return summary_text

    def _choose_next_slot(self, missing_slots: List[str], user_text: str, extracted_anything: bool) -> str:
        """Choose the next most logical slot to ask for."""
        if not missing_slots:
            return "summary"
        
        # Priority order for slots (matches conversation_flow)
        priority_order = [
            "pincode",
            "expected_salary",
            "has_two_wheeler", 
            "languages",
            "availability_date",
            "preferred_shift",
            "total_experience_months",
            "confirmation"
        ]
        
        # Return first missing slot in priority order
        for slot in priority_order:
            if slot in missing_slots:
                return slot
        
        return missing_slots[0]

    def _get_slot_description(self, slot: str) -> str:
        """Get user-friendly description for each slot."""
        descriptions = {
            "pincode": "pincode ya area",
            "availability_date": "kab se kaam shuru kar sakte hain",
            "preferred_shift": "preferred shift timing (morning, evening, ya night)",
            "expected_salary": "expected monthly salary",
            "languages": "kaunsi languages aati hain",
            "has_two_wheeler": "kya aapke paas two wheeler hai",
            "total_experience_months": "kitna experience hai (months mein)",
            "confirmation": "kya yeh information sahi hai?"
        }
        return descriptions.get(slot, slot.replace('_', ' '))

    def _get_candidate_summary(self, candidate: Candidate) -> str:
        """Get a summary of current candidate data."""
        data = []
        if candidate.pincode:
            data.append(f"Pincode: {candidate.pincode}")
        if candidate.locality:
            data.append(f"Area: {candidate.locality}")
        if candidate.availability_date:
            data.append(f"Availability: {candidate.availability_date}")
        if candidate.preferred_shift:
            data.append(f"Shift: {candidate.preferred_shift.value}")
        if candidate.expected_salary:
            data.append(f"Salary: ₹{candidate.expected_salary}")
        if candidate.languages:
            langs = [l.value for l in candidate.languages]
            if candidate.other_languages:
                langs.extend([f"Other:{ol}" for ol in candidate.other_languages])
            data.append(f"Languages: {', '.join(langs)}")
        if candidate.has_two_wheeler is not None:
            data.append(f"Two-wheeler: {'Yes' if candidate.has_two_wheeler else 'No'}")
        if candidate.total_experience_months is not None:
            data.append(f"Experience: {candidate.total_experience_months} months")
        
        return "; ".join(data) if data else "No data collected yet"

    def _try_update_slot(self, candidate: Candidate, state: ConversationState, slot: str, entities: Dict, text: str) -> bool:
        """Try to update a specific slot with extracted entities."""
        updated = False
        
        if slot == "pincode":
            updated = self._extract_and_set_pincode(candidate, entities, text)
        elif slot == "availability_date":
            updated = self._extract_and_set_availability(candidate, entities, text)
        elif slot == "preferred_shift":
            updated = self._extract_and_set_shift(candidate, entities, text)
        elif slot == "expected_salary":
            updated = self._extract_and_set_salary(candidate, entities, text)
        elif slot == "languages":
            updated = self._extract_and_set_languages(candidate, entities, text)
        elif slot == "has_two_wheeler":
            updated = self._extract_and_set_vehicle(candidate, entities, text)
        elif slot == "total_experience_months":
            updated = self._extract_and_set_experience(candidate, entities, text)
        
        if updated and slot not in state.fields_completed:
            state.fields_completed.append(slot)
            
        return updated
    
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
    
    def _extract_and_set_pincode(self, candidate: Candidate, entities: Dict, text: str) -> bool:
        """Extract and set pincode/locality."""
        value = entities.get("value") if isinstance(entities, dict) else None
        field = entities.get("field") if isinstance(entities, dict) else None
        method = entities.get("method") if isinstance(entities, dict) else None
        original_locality = entities.get("original_locality") if isinstance(entities, dict) else None
        
        if value:
            import re
            if isinstance(value, str) and re.fullmatch(r"\d{6}", value):
                candidate.pincode = value
                # If this was resolved from a locality, also store the original locality for reference
                if original_locality and method in ["locality_resolved", "llm_locality_resolved"]:
                    candidate.locality = original_locality
                    logger.info(f"Set pincode {value} (resolved from locality '{original_locality}')")
                else:
                    candidate.locality = None  # Clear locality when we have pincode
                return True
            # treat short textual value as locality
            if isinstance(value, str) and any(ch.isalpha() for ch in value):
                loc = value.strip().title()
                # Avoid generic acknowledgements becoming locality
                banned = {"Ok", "Okay", "Correct", "Sahi", "Haan", "Yes"}
                if loc not in banned and len(loc) >= 3:
                    candidate.locality = loc
                    candidate.pincode = None  # Clear pincode when we have locality
                    return True
            # if list provided by model, take first stringy token
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and any(ch.isalpha() for ch in v):
                        loc = v.strip().title()
                        banned = {"Ok", "Okay", "Correct", "Sahi", "Haan", "Yes"}
                        if loc not in banned and len(loc) >= 3:
                            candidate.locality = loc
                            candidate.pincode = None  # Clear pincode when we have locality
                            return True
        # Simple fallback - look for 6-digit number in raw text
        import re
        pincode_match = re.search(r'\b\d{6}\b', text)
        if pincode_match:
            candidate.pincode = pincode_match.group()
            return True
        # Fallbacks for locality from longer utterances
        # Heuristic: from tokens, remove fillers and take a plausible place token
        tokens = [t for t in re.findall(r"[A-Za-z\u0900-\u097F]+", text)]
        if tokens:
            # Filter out common filler/ack words
            banned_low = {"ok", "okay", "correct", "sahi", "haan", "yes", "main", "mein", "hoon", "rehti", "rahti", "rahati"}
            filtered = [t for t in tokens if t.lower() not in banned_low]
            if filtered:
                guess = filtered[-1]
                if len(guess) >= 3:
                    candidate.locality = guess.title()
                    return True
        # Fallback: if user spoke a short area name, accept as locality
        cleaned = text.strip()
        if any(ch.isalpha() for ch in cleaned) and len(cleaned.split()) <= 6:
            loc = cleaned.title()
            banned = {"Ok", "Okay", "Correct", "Sahi", "Haan", "Yes"}
            if loc not in banned and len(loc) >= 3:
                candidate.locality = loc
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
        """Extract and set language skills with negation and exclusivity handling.
        - Understand phrases like 'hindi nahi bol sakte' (exclude Hindi)
        - Understand 'sirf/only X' (exclusive selection)
        - Capture non-enum languages (e.g., French) into other_languages
        """
        value = entities.get("value") if isinstance(entities, dict) else None
        unknown_list = entities.get("unknown", []) if isinstance(entities, dict) else []
        text_lower = text.lower()

        # Base sets to accumulate
        positive_enums: List[LanguageSkill] = []
        positive_others: List[str] = []

        # Accept structured values from NLU if provided
        if isinstance(value, list):
            for lang in value:
                if isinstance(lang, str):
                    try:
                        enum_val = LanguageSkill(lang.lower())
                        if enum_val not in positive_enums:
                            positive_enums.append(enum_val)
                    except ValueError:
                        c = lang.strip().title()
                        if c and c not in positive_others:
                            positive_others.append(c)

        import re
        neg_terms = ["nahi", "nahin", "not", "cannot", "can't", "no", "नहीं"]
        exclusive_terms = ["sirf", "only", "keval", "केवल", "bas"]

        def is_negated(key: str) -> bool:
            window = 24
            patterns = [
                rf"{re.escape(key)}\W{{0,{window}}}(?:{'|'.join(map(re.escape, neg_terms))})",
                rf"(?:{'|'.join(map(re.escape, neg_terms))})\W{{0,{window}}}{re.escape(key)}",
            ]
            return any(re.search(p, text_lower) for p in patterns)

        def is_exclusive(key: str) -> bool:
            window = 24
            patterns = [
                rf"{re.escape(key)}\W{{0,{window}}}(?:{'|'.join(map(re.escape, exclusive_terms))})",
                rf"(?:{'|'.join(map(re.escape, exclusive_terms))})\W{{0,{window}}}{re.escape(key)}",
            ]
            return any(re.search(p, text_lower) for p in patterns)

        # Catalog of languages: map of phrase -> (enum or None, canonical display)
        catalog: Dict[str, Tuple[Optional[LanguageSkill], str]] = {
            # Enums (with synonyms)
            "hindi": (LanguageSkill.HINDI, "Hindi"),
            "हिंदी": (LanguageSkill.HINDI, "Hindi"),
            "english": (LanguageSkill.ENGLISH, "English"),
            "angrezi": (LanguageSkill.ENGLISH, "English"),
            "gujarati": (LanguageSkill.GUJARATI, "Gujarati"),
            "गुजराती": (LanguageSkill.GUJARATI, "Gujarati"),
            "marathi": (LanguageSkill.MARATHI, "Marathi"),
            "मराठी": (LanguageSkill.MARATHI, "Marathi"),
            # Common non-enum globals
            "french": (None, "French"),
            "german": (None, "German"),
            "spanish": (None, "Spanish"),
            "arabic": (None, "Arabic"),
            "urdu": (None, "Urdu"),
            "bengali": (None, "Bengali"),
            "tamil": (None, "Tamil"),
            "telugu": (None, "Telugu"),
            "kannada": (None, "Kannada"),
            "malayalam": (None, "Malayalam"),
            "punjabi": (None, "Punjabi"),
            "odia": (None, "Odia"),
            "assamese": (None, "Assamese"),
        }

        mentioned_enums: List[LanguageSkill] = []
        mentioned_others: List[str] = []
        exclusive_hit = False

        for key, (enum_val, display) in catalog.items():
            if key in text_lower:
                if is_negated(key):
                    # Explicitly negated; ensure it's not included
                    continue
                if enum_val is not None:
                    if enum_val not in mentioned_enums:
                        mentioned_enums.append(enum_val)
                else:
                    if display not in mentioned_others:
                        mentioned_others.append(display)
                if is_exclusive(key):
                    exclusive_hit = True

        # If exclusivity detected, override to only those mentioned here
        if exclusive_hit and (mentioned_enums or mentioned_others):
            positive_enums = mentioned_enums
            positive_others = mentioned_others
        else:
            # Merge mentions into positives
            for e in mentioned_enums:
                if e not in positive_enums:
                    positive_enums.append(e)
            for o in mentioned_others:
                if o not in positive_others:
                    positive_others.append(o)

        changed = False
        if positive_enums:
            candidate.languages = positive_enums
            changed = True
        if positive_others or unknown_list:
            # Merge and dedupe other_languages
            existing = set([ol.lower() for ol in (candidate.other_languages or [])])
            for o in positive_others:
                if o.lower() not in existing:
                    candidate.other_languages.append(o)
                    existing.add(o.lower())
            for u in unknown_list:
                if isinstance(u, str) and u.strip() and u.lower() not in existing:
                    candidate.other_languages.append(u.strip())
                    existing.add(u.strip().lower())
            if positive_others or unknown_list:
                changed = True

        return changed
    
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
        # Handle string values from NLU
        if isinstance(value, str) and value.isdigit():
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
        
        if total_months >= 0:
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
                  bot_response: str, start_time: float, status: str, turn_id: Optional[str] = None):
        """Log conversation turn for analytics."""
        total_latency_ms = int((time.time() - start_time) * 1000)
        turn = VoiceTurn(
            turn_id=turn_id or str(uuid.uuid4()),
            candidate_id=candidate_id,
            asr_text=user_text,
            asr_confidence=confidence,
            extracted_entities={},
            chosen_prompt=status,
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
            "last_server_latency_ms": latencies[-1] if latencies else 0.0,
        } 

    def _build_low_conf_confirmation(self, slot: str, heard_text: str) -> str:
        """Build a concise confirmation prompt using the heard text for low-confidence ASR.
        Args:
            slot: Current slot for which we are collecting info
            heard_text: Raw transcript heard from the user
        Returns:
            A Hinglish confirmation string
        """
        value = heard_text.strip().strip('\"\'').replace("\n", " ")
        # Map internal slot to confirm prompt key
        slot_to_confirm_key = {
            "pincode": "pincode_confirm",
            "availability_date": "availability_confirm",
            "preferred_shift": "shift_confirm",
            "expected_salary": "salary_confirm",
            "languages": "languages_confirm",
            "has_two_wheeler": "two_wheeler_confirm",
            "total_experience_months": "experience_confirm",
        }
        confirm_key = slot_to_confirm_key.get(slot)
        if confirm_key and confirm_key in self.prompts:
            return self.prompts[confirm_key].format(value=value)
        # Generic fallback
        return f"Maine aisa suna: '{value}'. Kya yahi sahi hai?" 

    def _is_affirmative(self, text: str) -> bool:
        """Return True if text is a simple affirmative like 'haan' or 'yes'.
        Args:
            text: User utterance
        Returns:
            True if we should treat it as confirmation
        """
        import re
        t = (text or "").strip().lower()
        # Only accept unambiguous short affirmations
        patterns = [r"\bhaan\b", r"\bhan\b", r"\byes\b"]
        return any(re.search(p, t) for p in patterns) 

    def _normalize_confirmation_value(self, slot: str, raw: str) -> str:
        """Extract a concise value for confirmation prompts based on slot.
        Args:
            slot: slot name
            raw: raw ASR text
        Returns:
            cleaned short value for confirmation
        """
        txt = (raw or "").strip()
        # For location-like slots, prefer a 6-digit pincode or a place token
        if slot in ("pincode",):
            import re
            m = re.search(r"\b\d{6}\b", txt)
            if m:
                return m.group()
        
        # Tokenize (supports Latin + Devanagari)
        import re
        tokens = re.findall(r"[A-Za-z\u0900-\u097F]+", txt)
        if not tokens:
            return txt
        
        # Remove common filler/stop-words in Hindi/Hinglish
        stopwords = {
            # Devanagari
            "में", "हूँ", "हूं", "है", "रहती", "रहता", "रहते", "मैं", "मे", "मुझे", "का", "की", "के",
            # Latin transliterations
            "mein", "hoon", "hain", "hai", "rehti", "rehta", "rehte", "main", "me", "mujhe", "ka", "ki", "ke",
            # English fillers
            "i", "am", "the", "a", "an", "to", "in", "on", "at"
        }
        filtered = [t for t in tokens if t.lower() not in stopwords]
        if not filtered:
            # If everything filtered, fall back to first token
            filtered = tokens
        
        # Heuristic per slot
        if slot == "expected_salary":
            digits = re.findall(r"\d+[\d,]*", txt)
            if digits:
                return digits[0]
        if slot == "total_experience_months":
            digits = re.findall(r"\d+", txt)
            if digits:
                return digits[0]
        if slot == "preferred_shift":
            # prefer the first non-stopword token
            return filtered[0]
        if slot == "languages":
            return filtered[0]
        if slot == "has_two_wheeler":
            return filtered[0]
        if slot == "availability_date":
            return filtered[0]
        
        # Default: choose the first meaningful token (place-like word comes first often)
        return filtered[0]

    async def _classify_affirmation_via_llm(
        self,
        user_text: str,
        history: List[Tuple[str, str]],
        slot: Optional[str] = None,
        proposed_value: Optional[str] = None,
    ) -> str:
        """Classify user's reply as yes/no using LLM; return 'yes' or 'no'."""
        if not user_text.strip():
            return "no"
        try:
            # Build compact history
            recent = []
            for spk, txt in history[-6:]:
                if txt:
                    recent.append(f"{spk}: {txt}")
            context = "\n".join(recent)
            # Compose instruction with slot/value context
            slot_desc = (slot or "").replace("_", " ")
            proposed = (proposed_value or "").strip()
            system = (
                "You are a yes/no confirmation classifier for a voice interview. "
                "Output exactly 'yes' or 'no' in lowercase with no punctuation and no explanations. "
                "Use the conversation context, the current slot, and proposed value to disambiguate short replies like 'haan', 'nahi', 'hmm'."
            )
            user = (
                f"CONTEXT (latest last):\n{context}\n\n"
                f"CURRENT SLOT: {slot_desc or '(unknown)'}\n"
                f"PROPOSED VALUE: {proposed or '(none)'}\n"
                f"USER REPLY: {user_text.strip()[:200]}\n\n"
                "Task: Is the user affirming the proposed value? Reply only 'yes' or 'no'."
            )
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                max_tokens=2,
                temperature=0.0,
            )
            out = (response.choices[0].message.content or "").strip().lower()
            return "yes" if out.startswith("y") else "no"
        except Exception as e:
            logger.error(f"Affirmation classification failed: {e}")
            return "no"

    def _apply_confirmed_value(self, candidate: Candidate, state: ConversationState, slot: str, value: Optional[str]) -> None:
        """Persist the pending confirmation value into the candidate profile for the given slot."""
        if not value:
            return
        v = (value or "").strip()
        if slot == "pincode":
            import re
            if re.fullmatch(r"\d{6}", v):
                candidate.pincode = v
            else:
                candidate.locality = v.title()
        elif slot == "availability_date":
            candidate.availability_date = v
        elif slot == "preferred_shift":
            try:
                candidate.preferred_shift = ShiftPreference(v.lower())
            except Exception:
                candidate.preferred_shift = ShiftPreference.FLEXIBLE
        elif slot == "expected_salary":
            try:
                candidate.expected_salary = int("".join([ch for ch in v if ch.isdigit()]))
            except Exception:
                pass
        elif slot == "languages":
            if v:
                candidate.other_languages = list({*(candidate.other_languages or []), v.title()})
        elif slot == "has_two_wheeler":
            candidate.has_two_wheeler = v.lower() in ("yes", "haan", "han", "true", "h")
        elif slot == "total_experience_months":
            try:
                candidate.total_experience_months = int("".join([ch for ch in v if ch.isdigit()]))
            except Exception:
                pass 