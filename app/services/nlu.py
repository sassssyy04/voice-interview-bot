import re
import asyncio
import time
from typing import Dict, Any, Optional, List, Tuple
try:
    import openai
except ImportError:
    openai = None
from app.core.config import settings
from app.core.logger import logger
from app.models.candidate import ShiftPreference, LanguageSkill


class NLUService:
    """Natural Language Understanding service for Hinglish entity extraction."""
    
    def __init__(self):
        if settings.openai_api_key:
            openai.api_key = settings.openai_api_key

    async def generate_contextual_response(
        self,
        user_text: str,
        history: List[Tuple[str, str]],
        candidate_snapshot: Dict[str, Any],
        current_field: str,
        next_field: Optional[str] = None,
    ) -> str:
        """Generate a short contextual response using LLM, considering history and current slot.

        Args:
            user_text: Latest user utterance
            history: List of (speaker, text) tuples, most recent last
            candidate_snapshot: Current candidate profile dict
            current_field: Slot currently being collected
            next_field: The next slot coming after the current one

        Returns:
            A concise Hinglish response string
        """
        # If no LLM configured, provide a simple fallback
        if not settings.openai_api_key or openai is None:
            if '?' in user_text:
                return "Aapne sawal poocha: '" + user_text.strip() + "'. Chaliye, pehle mera sawal complete karte hain: " + current_field.replace('_', ' ') + "."
            return "Samjha. Pehle is question ka jawab de dijiye: " + current_field.replace('_', ' ') + "."

        # Build a compact conversation context (last 6 turns)
        recent_lines = []
        for speaker, text in history[-6:]:
            if not text:
                continue
            recent_lines.append(f"{speaker}: {text}")
        context_block = "\n".join(recent_lines) if recent_lines else "(no prior turns)"

        current_label = current_field.replace('_', ' ')
        next_label = (next_field or '').replace('_', ' ') if next_field else None

        system_prompt = (
            "You are a polite Hinglish voice assistant for a job-screening flow.\n"
            "- Answer the user's off-topic question briefly and clearly in Hinglish if it helps them.\n"
            "- If they ask to repeat or clarify the last question, restate the last bot question naturally.\n"
            "- Then gently guide back to the current question (slot): '" + current_label + "'.\n"
            + ("- You may also mention that next we'll ask about '" + next_label + "'.\n" if next_label else "") +
            "- Keep responses under 2 short sentences. No markdown."
        )

        user_prompt = (
            "CONVERSATION SO FAR (latest last):\n" + context_block + "\n\n"
            "CURRENT SLOT: " + current_field + "\n"
            + ("NEXT SLOT: " + next_field + "\n" if next_field else "") +
            "CANDIDATE SNAPSHOT: " + str(candidate_snapshot) + "\n\n"
            "USER SAID: " + user_text + "\n\n"
            "Task: Respond in Hinglish, answer the user's query if reasonable, and guide back to the current question."
        )

        try:
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                max_tokens=120,
                temperature=0.3,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"Contextual LLM response failed: {e}")
            return "Samjha. Pehle is question ka jawab de dijiye: " + current_label + "."
    
    async def extract_entities(self, text: str, field: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Extract entities from Hinglish text for a specific field with validation.
        
        Args:
            text (str): User input text
            field (str): Field being collected (pincode, salary, etc.)
            context (Dict[str, Any]): Conversation context
            
        Returns:
            Dict[str, Any]: Extracted entities and metadata
        """
        start_time = time.time()
        
        # Normalize text first
        normalized_text = self._normalize_text(text)
        
        # Use appropriate extraction method based on field
        if field == "pincode":
            result = await self._extract_pincode(normalized_text)
        elif field == "availability_date":
            result = await self._extract_availability(normalized_text)
        elif field == "preferred_shift":
            result = await self._extract_shift(normalized_text)
        elif field == "expected_salary":
            result = await self._extract_salary(normalized_text)
        elif field == "languages":
            result = await self._extract_languages(normalized_text)
        elif field == "has_two_wheeler":
            result = await self._extract_boolean(normalized_text, field)
        elif field == "total_experience_months":
            result = await self._extract_experience(normalized_text)
        else:
            # Generic extraction using OpenAI
            result = await self._extract_with_openai(normalized_text, field, context)
        
        # Validate and enhance result
        result = self._validate_extraction_result(result, field, text)
        
        processing_time = (time.time() - start_time) * 1000
        
        logger.bind(metrics=True).info({
            "event": "nlu_extraction",
            "field": field,
            "input_text": text,
            "extracted_entities": result,
            "processing_time_ms": processing_time
        })
        
        return result
    
    def _validate_extraction_result(self, result: Dict[str, Any], field: str, original_text: str) -> Dict[str, Any]:
        """Validate and enhance extraction results with field-specific checks."""
        if not isinstance(result, dict) or not result.get("value"):
            return result
        
        value = result["value"]
        confidence = result.get("confidence", 0.0)
        
        # Field-specific validation
        if field == "expected_salary":
            if isinstance(value, (int, float)):
                # Validate salary range
                if not (1000 <= value <= 1000000):  # Extended range for validation
                    result["confidence"] = max(0.0, confidence - 0.3)
                    logger.warning(f"Salary {value} outside expected range")
                
        elif field == "total_experience_months":
            if isinstance(value, (int, float)):
                # Validate experience range
                if value < 0 or value > 600:  # 0 to 50 years max
                    result["confidence"] = max(0.0, confidence - 0.4)
                    logger.warning(f"Experience {value} months outside expected range")
                    
        elif field == "pincode":
            if isinstance(value, str):
                # Validate pincode format
                if not re.match(r'^\d{6}$', value):
                    result["confidence"] = max(0.0, confidence - 0.2)
                else:
                    # Check valid Indian pincode range
                    pincode_int = int(value)
                    if not (100000 <= pincode_int <= 999999):
                        result["confidence"] = max(0.0, confidence - 0.3)
                        
        elif field == "languages":
            if isinstance(value, list):
                # Validate language list isn't too long (suspicious)
                if len(value) > 5:
                    result["confidence"] = max(0.0, confidence - 0.2)
                    logger.warning(f"Unusually many languages detected: {len(value)}")
                    
        elif field == "preferred_shift":
            if isinstance(value, str):
                valid_shifts = {"morning", "afternoon", "evening", "night", "flexible"}
                if value.lower() not in valid_shifts:
                    result["confidence"] = max(0.0, confidence - 0.3)
                    logger.warning(f"Invalid shift value: {value}")
        
        # Generic validation - check if extracted value makes sense given input
        if confidence > 0.5 and len(original_text.strip()) < 3:
            # Very short input with high confidence extraction is suspicious
            result["confidence"] = max(0.0, confidence - 0.2)
            logger.warning(f"Short input '{original_text}' with high confidence extraction")
        
        # Add extraction quality metadata
        result["validation"] = {
            "original_confidence": confidence,
            "adjusted_confidence": result["confidence"],
            "validation_applied": True
        }
        
        return result
    
    async def generate_entity_confirmation_summary(self, candidate_data: Dict[str, Any]) -> str:
        """Generate a human-readable summary of all extracted entities for confirmation."""
        try:
            logger.info(f"Generating confirmation summary for candidate_data: {candidate_data}")
            
            # Validate input data
            if not isinstance(candidate_data, dict):
                logger.error(f"Invalid candidate_data type: {type(candidate_data)}")
                candidate_data = {}
                
            summary_parts = ["Dhanyawad! Main aapki saari details repeat karta hun:"]
            
            # Helper to format values nicely
            def format_value(value, field_type):
                if value is None:
                    return None
                
                if field_type == "languages":
                    if isinstance(value, list):
                        if len(value) == 0:
                            return None
                        elif len(value) == 1:
                            return value[0].title()
                        elif len(value) == 2:
                            return f"{value[0].title()} aur {value[1].title()}"
                        else:
                            return f"{', '.join(v.title() for v in value[:-1])} aur {value[-1].title()}"
                    return str(value).title()
                
                elif field_type == "salary":
                    if isinstance(value, (int, float)):
                        return f"₹{int(value):,} per month"
                    return str(value)
                
                elif field_type == "experience":
                    if isinstance(value, (int, float)):
                        months = int(value)
                        if months == 0:
                            return "Fresher (0 experience)"
                        elif months < 12:
                            return f"{months} month{'s' if months > 1 else ''}"
                        else:
                            years = months // 12
                            remaining_months = months % 12
                            if remaining_months == 0:
                                return f"{years} year{'s' if years > 1 else ''}"
                            else:
                                return f"{years} year{'s' if years > 1 else ''} {remaining_months} month{'s' if remaining_months > 1 else ''}"
                    return str(value)
                
                elif field_type == "shift":
                    if hasattr(value, 'value'):  # Handle enum
                        return value.value.replace('_', ' ').title()
                    elif isinstance(value, str):
                        return value.replace('_', ' ').title()
                    return str(value)
                
                elif field_type == "boolean":
                    if isinstance(value, bool):
                        return "Haan" if value else "Nahi"
                    return str(value)
                
                return str(value).title() if isinstance(value, str) else str(value)
            
            # Add each field if present
            if candidate_data.get('pincode'):
                pincode_text = f"• Pincode: {candidate_data['pincode']}"
                # If we also have locality, show it was resolved
                if candidate_data.get('locality'):
                    pincode_text += f" (from {candidate_data['locality']})"
                summary_parts.append(pincode_text)
            elif candidate_data.get('locality'):
                summary_parts.append(f"• Area: {candidate_data['locality']}")
            
            if candidate_data.get('expected_salary'):
                formatted_salary = format_value(candidate_data['expected_salary'], 'salary')
                summary_parts.append(f"• Expected salary: {formatted_salary}")
            
            if candidate_data.get('total_experience_months') is not None:
                formatted_exp = format_value(candidate_data['total_experience_months'], 'experience')
                summary_parts.append(f"• Experience: {formatted_exp}")
            
            if candidate_data.get('languages'):
                languages = candidate_data['languages']
                other_languages = candidate_data.get('other_languages', [])
                all_langs = []
                if isinstance(languages, list):
                    all_langs.extend([lang.value if hasattr(lang, 'value') else str(lang) for lang in languages])
                if other_languages:
                    all_langs.extend(other_languages)
                if all_langs:
                    formatted_langs = format_value(all_langs, 'languages')
                    summary_parts.append(f"• Languages: {formatted_langs}")
            
            if candidate_data.get('availability_date'):
                summary_parts.append(f"• Availability: {candidate_data['availability_date']}")
            
            if candidate_data.get('preferred_shift'):
                formatted_shift = format_value(candidate_data['preferred_shift'], 'shift')
                summary_parts.append(f"• Preferred shift: {formatted_shift}")
            
            if candidate_data.get('has_two_wheeler') is not None:
                formatted_vehicle = format_value(candidate_data['has_two_wheeler'], 'boolean')
                summary_parts.append(f"• Two wheeler: {formatted_vehicle}")
            
            # Add confirmation question
            summary_parts.append("\nKya yeh saari information sahi hai? Agar kuch galat hai ya change karna hai toh bataiye. Agar sab theek hai toh 'haan' ya 'correct' boliye.")
            
            result = "\n".join(summary_parts)
            logger.info(f"Generated confirmation summary: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Error generating entity confirmation summary: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            # Fallback summary
            return "Dhanyawad! Kya aapki saari details sahi hain? Haan ya nahi mein jawab dijiye."
    
    async def process_entity_confirmation_response(
        self, 
        user_response: str, 
        current_entities: Dict[str, Any],
        confidence: float = 1.0
    ) -> Dict[str, Any]:
        """Process user's response to entity confirmation using LLM to detect needed changes.
        
        Args:
            user_response: User's response to the entity confirmation
            current_entities: Dict of currently extracted entities
            confidence: ASR confidence score
            
        Returns:
            Dict containing:
            - action: "accept", "modify", or "clarify"
            - changes: Dict of fields that need to be changed (if action is "modify")
            - confirmation_text: Response text for user
            - needs_reconfirmation: Whether to ask for confirmation again
        """
        if not user_response.strip():
            return {
                "action": "clarify",
                "changes": {},
                "confirmation_text": "Main aapki awaaz sahi se nahi sun paaya. Dobara boliye - kya information sahi hai?",
                "needs_reconfirmation": True
            }
        
        # Quick check for simple acceptance
        acceptance_phrases = ["haan", "han", "yes", "correct", "sahi", "theek", "ok", "okay", "bilkul sahi"]
        if any(phrase in user_response.lower() for phrase in acceptance_phrases) and len(user_response.strip().split()) <= 2:
            return {
                "action": "accept",
                "changes": {},
                "confirmation_text": "Perfect! Saari details confirm ho gayi. Main aapke liye best jobs dhundta hun.",
                "needs_reconfirmation": False
            }
        
        # Use LLM for complex change detection
        if not settings.openai_api_key or openai is None:
            # Fallback without LLM
            if confidence < 0.3:
                return {
                    "action": "clarify", 
                    "changes": {},
                    "confirmation_text": "Awaaz clear nahi thi. Kya sab information sahi hai? Haan ya nahi mein jawab dijiye.",
                    "needs_reconfirmation": True
                }
            else:
                return {
                    "action": "accept",
                    "changes": {},
                    "confirmation_text": "Samjha, saari details confirm kar raha hun.",
                    "needs_reconfirmation": False
                }
        
        try:
            # Prepare entity summary for LLM
            entity_summary = self._prepare_entity_summary_for_llm(current_entities)
            
            system_prompt = (
                "You are processing a user's response to a job interview entity confirmation in Hinglish/English.\n"
                "The user was presented with their extracted information and asked if it's correct.\n"
                "Analyze their response and determine:\n"
                "1. Do they accept all information as correct?\n"
                "2. Do they want to change specific fields?\n"
                "3. Is their response unclear and needs clarification?\n\n"
                "Return ONLY a JSON object with these keys:\n"
                "- action: 'accept', 'modify', or 'clarify'\n"
                "- changes: object with field names as keys and new values\n"
                "- reason: brief explanation of decision\n\n"
                "Available fields to modify: pincode, locality, expected_salary, total_experience_months, "
                "languages, availability_date, preferred_shift, has_two_wheeler"
            )
            
            user_prompt = (
                f"CURRENT EXTRACTED ENTITIES:\n{entity_summary}\n\n"
                f"USER RESPONSE: '{user_response}'\n\n"
                f"ASR CONFIDENCE: {confidence}\n\n"
                "Examples:\n"
                "- 'haan sab theek hai' → action: accept\n"
                "- 'salary galat hai, 25000 chahiye' → action: modify, changes: {\"expected_salary\": 25000}\n"
                "- 'area change karna hai' → action: modify, changes: {\"locality\": \"<need_clarification>\"}\n"
                "- unclear/garbled → action: clarify\n\n"
                "Analyze the response and return JSON:"
            )
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=200,
                temperature=0.1
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                result_data = json.loads(content[start:end])
                
                action = result_data.get('action', 'clarify')
                changes = result_data.get('changes', {})
                reason = result_data.get('reason', '')
                
                # Generate appropriate response text
                if action == 'accept':
                    confirmation_text = "Perfect! Saari details confirm ho gayi. Main aapke liye best jobs dhundta hun."
                    needs_reconfirmation = False
                    
                elif action == 'modify':
                    if changes:
                        # Apply changes and generate confirmation
                        modified_fields = []
                        for field, new_value in changes.items():
                            if new_value == "<need_clarification>":
                                field_desc = self._get_field_description(field)
                                return {
                                    "action": "clarify",
                                    "changes": {field: None},
                                    "confirmation_text": f"Samjha, aap {field_desc} change karna chahte hain. Nayi {field_desc} bataiye.",
                                    "needs_reconfirmation": True
                                }
                            else:
                                modified_fields.append(self._get_field_description(field))
                        
                        confirmation_text = f"Samjha, main {', '.join(modified_fields)} update kar diya. Kya ab sab theek hai?"
                        needs_reconfirmation = True
                    else:
                        confirmation_text = "Kya change karna hai? Specific field bataiye jaise salary, area, experience."
                        needs_reconfirmation = True
                        
                else:  # clarify
                    confirmation_text = "Samajh nahi aaya. Kya saari information sahi hai? Ya kuch change karna hai?"
                    needs_reconfirmation = True
                
                return {
                    "action": action,
                    "changes": changes,
                    "confirmation_text": confirmation_text,
                    "needs_reconfirmation": needs_reconfirmation,
                    "llm_reason": reason
                }
                
        except Exception as e:
            logger.error(f"LLM entity confirmation processing failed: {e}")
            
        # Fallback processing
        return {
            "action": "clarify",
            "changes": {},
            "confirmation_text": "Main samjha nahi. Kya saari details sahi hai? Haan ya nahi mein jawab dijiye.",
            "needs_reconfirmation": True
        }
    
    def _prepare_entity_summary_for_llm(self, entities: Dict[str, Any]) -> str:
        """Prepare a clean summary of entities for LLM processing."""
        summary_lines = []
        
        for field, value in entities.items():
            if value is not None:
                if field == 'languages':
                    if isinstance(value, list) and value:
                        lang_str = ', '.join([str(v) for v in value])
                        summary_lines.append(f"Languages: {lang_str}")
                elif field == 'expected_salary':
                    summary_lines.append(f"Expected Salary: ₹{value}")
                elif field == 'total_experience_months':
                    years = value // 12
                    months = value % 12
                    if years > 0:
                        exp_str = f"{years} years" + (f" {months} months" if months > 0 else "")
                    else:
                        exp_str = f"{months} months"
                    summary_lines.append(f"Experience: {exp_str}")
                elif field == 'has_two_wheeler':
                    summary_lines.append(f"Two Wheeler: {'Yes' if value else 'No'}")
                elif field == 'preferred_shift':
                    shift_name = str(value).replace('_', ' ').title()
                    summary_lines.append(f"Preferred Shift: {shift_name}")
                else:
                    summary_lines.append(f"{field.replace('_', ' ').title()}: {value}")
        
        return '\n'.join(summary_lines)
    
    def _get_field_description(self, field: str) -> str:
        """Get user-friendly description for field names."""
        descriptions = {
            "pincode": "pincode",
            "locality": "area",
            "expected_salary": "salary",
            "total_experience_months": "experience",
            "languages": "languages",
            "availability_date": "availability date",
            "preferred_shift": "shift timing",
            "has_two_wheeler": "two wheeler information"
        }
        return descriptions.get(field, field.replace('_', ' '))
    
    async def apply_entity_changes(
        self, 
        current_entities: Dict[str, Any], 
        changes: Dict[str, Any],
        user_response: str = ""
    ) -> Dict[str, Any]:
        """Apply detected changes to current entities, with re-extraction for new values.
        
        Args:
            current_entities: Current entity values
            changes: Dict of changes detected by LLM
            user_response: Original user response (for re-extraction if needed)
            
        Returns:
            Updated entities dict
        """
        updated_entities = current_entities.copy()
        
        for field, new_value in changes.items():
            try:
                if new_value is None or new_value == "<need_clarification>":
                    # Clear the field - will be re-asked
                    if field in updated_entities:
                        updated_entities[field] = None
                    continue
                
                # Apply field-specific processing
                if field == "expected_salary":
                    # Parse salary values
                    if isinstance(new_value, str):
                        # Extract number from string like "25000" or "25 thousand"
                        salary_result = await self._extract_salary(new_value)
                        if salary_result.get("value"):
                            updated_entities[field] = salary_result["value"]
                        else:
                            updated_entities[field] = self._parse_salary_from_text(new_value)
                    else:
                        updated_entities[field] = int(new_value) if new_value else None
                        
                elif field == "total_experience_months":
                    # Parse experience values
                    if isinstance(new_value, str):
                        exp_result = await self._extract_experience(new_value)
                        if exp_result.get("value") is not None:
                            updated_entities[field] = exp_result["value"]
                        else:
                            updated_entities[field] = self._parse_experience_from_text(new_value)
                    else:
                        updated_entities[field] = int(new_value) if new_value else None
                        
                elif field == "languages":
                    # Parse language values
                    if isinstance(new_value, str):
                        lang_result = await self._extract_languages(new_value)
                        if lang_result.get("value"):
                            updated_entities[field] = lang_result["value"]
                            if lang_result.get("unknown"):
                                updated_entities["other_languages"] = lang_result["unknown"]
                        else:
                            # Simple parsing for direct language names
                            updated_entities[field] = self._parse_languages_from_text(new_value)
                    elif isinstance(new_value, list):
                        updated_entities[field] = new_value
                    else:
                        updated_entities[field] = [str(new_value)] if new_value else []
                        
                elif field == "has_two_wheeler":
                    # Parse boolean values
                    if isinstance(new_value, str):
                        bool_result = await self._extract_boolean(new_value, field)
                        updated_entities[field] = bool_result.get("value")
                    else:
                        updated_entities[field] = bool(new_value)
                        
                elif field == "preferred_shift":
                    # Parse shift values
                    if isinstance(new_value, str):
                        shift_result = await self._extract_shift(new_value)
                        if shift_result.get("value"):
                            updated_entities[field] = shift_result["value"]
                        else:
                            # Direct mapping for common values
                            shift_map = {
                                "morning": "morning", "subah": "morning",
                                "afternoon": "afternoon", "dopahar": "afternoon", 
                                "evening": "evening", "shaam": "evening",
                                "night": "night", "raat": "night",
                                "flexible": "flexible", "any": "flexible"
                            }
                            updated_entities[field] = shift_map.get(new_value.lower(), new_value)
                    else:
                        updated_entities[field] = str(new_value) if new_value else None
                        
                elif field in ["pincode", "locality"]:
                    # Parse location values
                    if isinstance(new_value, str):
                        loc_result = await self._extract_pincode(new_value)
                        if loc_result.get("value"):
                            if loc_result.get("field") == "pincode":
                                updated_entities["pincode"] = loc_result["value"]
                                updated_entities["locality"] = None
                            else:
                                updated_entities["locality"] = loc_result["value"]
                                updated_entities["pincode"] = None
                        else:
                            updated_entities[field] = new_value.title()
                    else:
                        updated_entities[field] = str(new_value) if new_value else None
                        
                else:
                    # Direct assignment for other fields
                    updated_entities[field] = new_value
                    
            except Exception as e:
                logger.error(f"Error applying change to {field}: {e}")
                # Keep original value on error
                continue
        
        return updated_entities
    
    def _parse_salary_from_text(self, text: str) -> Optional[int]:
        """Simple salary parsing for direct updates."""
        import re
        if not text:
            return None
        
        # Extract numbers
        numbers = re.findall(r'\d+', str(text))
        if not numbers:
            return None
        
        amount = int(numbers[0])
        text_lower = str(text).lower()
        
        # Apply multipliers
        if any(word in text_lower for word in ['lakh', 'lac']):
            amount *= 100000
        elif any(word in text_lower for word in ['thousand', 'hazaar', 'k']):
            amount *= 1000
            
        return amount if 1000 <= amount <= 1000000 else None
    
    def _parse_experience_from_text(self, text: str) -> Optional[int]:
        """Simple experience parsing for direct updates."""
        import re
        if not text:
            return None
            
        numbers = re.findall(r'\d+', str(text))
        if not numbers:
            return None
            
        value = int(numbers[0])
        text_lower = str(text).lower()
        
        # Convert to months
        if any(word in text_lower for word in ['year', 'saal', 'years']):
            return value * 12
        else:
            return value  # Assume months
    
    def _parse_languages_from_text(self, text: str) -> List[str]:
        """Simple language parsing for direct updates."""
        if not text:
            return []
            
        text_lower = str(text).lower()
        languages = []
        
        # Check for common languages
        lang_map = {
            'hindi': 'hindi', 'english': 'english', 'marathi': 'marathi',
            'bengali': 'bengali', 'tamil': 'tamil', 'telugu': 'telugu',
            'gujarati': 'gujarati', 'kannada': 'kannada', 'punjabi': 'punjabi'
        }
        
        for lang_key, lang_value in lang_map.items():
            if lang_key in text_lower:
                languages.append(lang_value)
                
        return languages
    
    async def get_pincode_from_city_llm(self, city_name: str) -> Optional[str]:
        """Get a representative pincode for any Indian city/locality using LLM."""
        if not city_name or not settings.openai_api_key or openai is None:
            return None
            
        try:
            prompt = (
                "You are a comprehensive Indian geography database. Given a city, town, locality, or area name in India, "
                "return the most representative/central pincode for that location.\n\n"
                "Rules:\n"
                "- For major cities, use the main/central pincode (e.g., Mumbai → 400001, Delhi → 110001)\n"
                "- For specific areas within cities, use the area's pincode if known\n"
                "- For smaller towns, use their main pincode\n"
                "- If you're not certain about a location, return null\n"
                "- Handle common variations (e.g., Bengaluru/Bangalore, Bombay/Mumbai)\n\n"
                "Return ONLY a JSON object: {\"pincode\": \"XXXXXX\"} or {\"pincode\": null}\n\n"
                f"Location: '{city_name}'\n"
                "Response:"
            )
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=100,
                temperature=0.0
            )
            
            content = response.choices[0].message.content.strip()
            
            # Extract JSON from response
            import json
            if '{' in content and '}' in content:
                start = content.find('{')
                end = content.rfind('}') + 1
                data = json.loads(content[start:end])
                pincode = data.get('pincode')
                
                # Validate pincode format
                if pincode and isinstance(pincode, str) and re.match(r'^\d{6}$', pincode):
                    pincode_int = int(pincode)
                    # Validate Indian pincode range
                    if 100000 <= pincode_int <= 999999:
                        logger.info(f"Resolved '{city_name}' to pincode '{pincode}' via LLM")
                        return pincode
                        
        except Exception as e:
            logger.error(f"LLM pincode resolution failed for '{city_name}': {e}")
        
        return None
    
    async def get_pincode_from_city_web(self, city_name: str) -> Optional[str]:
        """Get pincode for a city using web search as fallback."""
        if not city_name:
            return None
            
        # Only try web search if we have WebSearch available
        try:
            # Use WebSearch tool if available
            search_query = f"{city_name} pincode India postal code"
            
            # This would use the WebSearch tool - but we'll keep it simple for now
            # and rely on the LLM which has comprehensive geographic knowledge
            logger.info(f"Web search fallback not implemented, skipping for '{city_name}'")
            return None
            
        except Exception as e:
            logger.error(f"Web search pincode resolution failed for '{city_name}': {e}")
            return None
    
    async def resolve_pincode_from_locality(self, locality: str) -> Optional[str]:
        """Resolve pincode from locality using dynamic LLM-based resolution."""
        if not locality:
            return None
            
        # Primary method: Use LLM for comprehensive city/locality resolution
        pincode = await self.get_pincode_from_city_llm(locality)
        if pincode:
            return pincode
        
        # Fallback: Try web search (if implemented)
        pincode = await self.get_pincode_from_city_web(locality)
        if pincode:
            return pincode
        
        logger.warning(f"Could not resolve pincode for locality: '{locality}'")
        return None
    
    def _normalize_text(self, text: str) -> str:
        """Normalize text for entity extraction."""
        if not text:
            return text
            
        # Convert to lowercase
        text = text.lower().strip()
        
        # Devanagari to English normalizations (for ElevenLabs ASR output)
        devanagari_normalizations = {
            # Numbers in Devanagari
            "एक": "1", "दो": "2", "तीन": "3", "चार": "4", "पांच": "5", "पाँच": "5",
            "छह": "6", "छः": "6", "सात": "7", "आठ": "8", "नौ": "9", "दस": "10",
            "ग्यारह": "11", "बारह": "12", "तेरह": "13", "चौदह": "14", "पंद्रह": "15", "पन्द्रह": "15",
            "सोलह": "16", "सत्रह": "17", "अठारह": "18", "उन्नीस": "19", "बीस": "20",
            "हजार": "thousand", "हज़ार": "thousand", "लाख": "lakh", "करोड़": "crore",
            
            # Common Hindi words for entity extraction
            "हाँ": "yes", "हां": "yes", "जी हाँ": "yes", "जी हां": "yes", "बिल्कुल": "yes",
            "नहीं": "no", "नही": "no", "ना": "no", "बिल्कुल नहीं": "no",
            
            # Time/Date expressions
            "आज": "today", "कल": "tomorrow", "परसों": "day after tomorrow",
            "अभी": "now", "तुरंत": "immediately", "जल्दी": "soon", "इमीडिएटली": "immediately",
            
            # Location/Place words
            "में": "in", "पर": "at", "से": "from", "तक": "to", "का": "of", "की": "of", "के": "of",
            "रहता": "live", "रहती": "live", "रहते": "live", "घर": "home", "जगह": "place",
            "एरिया": "area", "इलाका": "area", "पिन": "pin", "कोड": "code", "पिनकोड": "pincode",
            "दिल्ली": "delhi", "मुंबई": "mumbai", "बैंगलोर": "bangalore", "पुणे": "pune", "चेन्नई": "chennai",
            
            # Experience/Work
            "साल": "years", "सालों": "years", "महीना": "months", "महीने": "months", "महीनों": "months",
            "काम": "work", "कार्य": "work", "अनुभव": "experience", "एक्सपीरियंस": "experience",
            "वेयरहाउस": "warehouse", "फैक्ट्री": "factory", "ऑफिस": "office", "दुकान": "shop",
            
            # Money/Salary
            "रुपए": "rupees", "रुपये": "rupees", "रुपया": "rupee", "पैसा": "money", "पैसे": "money",
            "सैलरी": "salary", "तनख्वाह": "salary", "वेतन": "salary", "एक्सपेक्ट": "expect",
            "चाहिए": "need", "होगा": "will", "करता": "do", "हूं": "am", "हूँ": "am",
            "पर": "per", "मंथ": "month", "महीना": "month", "थाउजेंड": "thousand", "थाउज़ेंड": "thousand",
            "फिफ्टीन": "fifteen", "ट्वेंटी": "twenty", "थर्टी": "thirty", "फोर्टी": "forty", "फिफ्टी": "fifty",
            
            # Vehicle related
            "बाइक": "bike", "स्कूटर": "scooter", "मोटरसाइकिल": "motorcycle", "साइकिल": "bicycle",
            "गाड़ी": "vehicle", "पास": "have", "है": "is", "हैं": "are", "मेरे": "my",
            "बजाज": "bajaj", "हीरो": "hero", "होंडा": "honda", "यामाहा": "yamaha", "पुलसर": "pulsar",
            
            # Language related
            "भाषा": "language", "बोल": "speak", "बोलता": "speak", "बोलती": "speak", "बोलते": "speak",
            "सकता": "can", "सकती": "can", "सकते": "can", "हिंदी": "hindi", "अंग्रेजी": "english",
            "इंग्लिश": "english", "मराठी": "marathi", "तमिल": "tamil", "तेलुगु": "telugu",
            "दोनों": "both", "और": "and", "भी": "also", "थोड़ा": "little", "अच्छा": "good",
            
            # Shift/Time related
            "शिफ्ट": "shift", "टाइम": "time", "समय": "time", "सुबह": "morning", "शाम": "evening",
            "रात": "night", "मॉर्निंग": "morning", "इवनिंग": "evening", "नाइट": "night",
            "प्रेफर": "prefer", "पसंद": "prefer", "अच्छा": "good", "लगता": "feel", "लगती": "feel",
            
            # Start/Availability
            "शुरू": "start", "स्टार्ट": "start", "कर": "do", "सकता": "can", "सकती": "can",
            "तैयार": "ready", "उपलब्ध": "available"
        }
        
        # Apply Devanagari normalizations first
        for hindi, english in devanagari_normalizations.items():
            text = text.replace(hindi, english)
        
        # Common Hindi to English normalizations (Romanized)
        normalizations = {
            # Numbers
            "ek": "1", "do": "2", "teen": "3", "char": "4", "panch": "5",
            "che": "6", "saat": "7", "aath": "8", "nau": "9", "das": "10",
            "gyarah": "11", "barah": "12", "terah": "13", "chaudah": "14", "pandrah": "15",
            "solah": "16", "satrah": "17", "atharah": "18", "unnis": "19", "bees": "20",
            "hazaar": "thousand", "lakh": "lakh",
            
            # Time/Date
            "aaj": "today", "kal": "tomorrow", "parso": "day after tomorrow",
            "abhi": "now", "turant": "immediately", "jaldi": "soon",
            
            # Boolean
            "haan": "yes", "han": "yes", "ji haan": "yes", "bilkul": "yes",
            "nahi": "no", "nahin": "no", "na": "no", "bilkul nahi": "no",
            
            # Experience mishear fixes
            "saal": "years", "sal": "years",
            "yer": "years", "yers": "years", "yr": "years", "yrs": "years", "y": "years",
            "ear": "years", "ears": "years", "yar": "years", "yars": "years",
            "mahina": "months", "mahine": "months",
            "mnth": "months", "mnths": "months", "mth": "months", "mths": "months",
            "mon": "months", "mons": "months"
        }
        
        # English number words to digits (helps when ASR outputs 'two years')
        english_numbers = {
            "zero": "0", "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
            "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
            "eleven": "11", "twelve": "12", "thirteen": "13", "fourteen": "14", "fifteen": "15",
            "sixteen": "16", "seventeen": "17", "eighteen": "18", "nineteen": "19", "twenty": "20",
            "half": "0.5"
        }
        normalizations.update(english_numbers)
        
        for hindi, english in normalizations.items():
            text = re.sub(rf"\b{re.escape(hindi)}\b", english, text)
        
        # Clean up extra spaces
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    async def _extract_pincode(self, text: str) -> Dict[str, Any]:
        """Extract pincode or locality from text with improved validation."""
        text_lower = text.lower().strip()
        
        # Look for 6-digit numbers first (highest priority)
        pincode_pattern = r'\b(\d{6})\b'
        matches = re.findall(pincode_pattern, text)
        
        for match in matches:
            # Validate pincode ranges (Indian postal codes)
            pincode_int = int(match)
            if 100000 <= pincode_int <= 999999:  # Valid Indian pincode range
                return {
                    "value": match,
                    "confidence": 0.95,
                    "field": "pincode", 
                    "method": "regex"
                }
        
        # If no valid pincode, try to extract locality/area name
        # Remove common filler words
        filler_words = {'main', 'mein', 'me', 'hoon', 'hun', 'rehta', 'rehti', 'rahata', 'rahati', 
                       'live', 'in', 'at', 'from', 'near', 'se', 'mein', 'ke', 'ki', 'ka'}
        
        # Extract potential place names (alphabetic tokens)
        tokens = re.findall(r'[A-Za-z\u0900-\u097F]{3,}', text)
        place_tokens = [token for token in tokens if token.lower() not in filler_words]
        
        # Look for known city patterns or area names
        if place_tokens:
            # Take the longest or last meaningful token as locality
            locality = max(place_tokens, key=len) if len(place_tokens) > 1 else place_tokens[0]
            
            # Additional validation - avoid generic responses
            invalid_localities = {'ok', 'okay', 'correct', 'sahi', 'haan', 'yes', 'right', 'samjha'}
            if locality.lower() not in invalid_localities and len(locality) >= 3:
                # Try to resolve locality to pincode for better distance matching
                resolved_pincode = await self.resolve_pincode_from_locality(locality)
                if resolved_pincode:
                    return {
                        "value": resolved_pincode,
                        "confidence": 0.8,
                        "field": "pincode",
                        "method": "locality_resolved",
                        "original_locality": locality.title()
                    }
                else:
                    return {
                        "value": locality.title(),
                        "confidence": 0.7,
                        "field": "locality",
                        "method": "rule"
                    }
        
        # Try LLM for complex location extraction
        if settings.openai_api_key and openai is not None:
            try:
                prompt = (
                    "Extract location (pincode or city/area name) from user text. Return ONLY a JSON object.\n"
                    "Rules:\n"
                    "- If 6-digit number present, return as pincode\n"
                    "- Otherwise extract city/area name\n"
                    "- Ignore filler words like 'main', 'mein', 'rehta', 'live'\n"
                    "- Return proper capitalized place names\n"
                    f"TEXT: '{text}'\n"
                    "Return: {\"type\": \"pincode\"|\"locality\", \"value\": \"<location>\"}"
                )
                resp = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0
                )
                content = resp.choices[0].message.content.strip()
                
                # Extract JSON
                import json as _json
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    data = _json.loads(content[start:end])
                    location_type = data.get('type', 'locality')
                    value = data.get('value', '').strip()
                    
                    if value and len(value) >= 3:
                        if location_type == 'pincode' and re.match(r'^\d{6}$', value):
                            return {
                                "value": value,
                                "confidence": 0.8,
                                "field": "pincode",
                                "method": "llm"
                            }
                        else:
                            # Try to resolve locality to pincode for better matching
                            resolved_pincode = await self.resolve_pincode_from_locality(value)
                            if resolved_pincode:
                                return {
                                    "value": resolved_pincode,
                                    "confidence": 0.8,
                                    "field": "pincode",
                                    "method": "llm_locality_resolved",
                                    "original_locality": value.title()
                                }
                            else:
                                return {
                                    "value": value.title(),
                                    "confidence": 0.8,
                                    "field": "locality", 
                                    "method": "llm"
                                }
            except Exception as e:
                logger.error(f"LLM pincode extraction failed: {e}")
        
        # Fallback to OpenAI generic extraction
        return await self._extract_with_openai(text, "pincode")
    
    async def _extract_availability(self, text: str) -> Dict[str, Any]:
        """Extract availability date from text."""
        # Check for common patterns
        if any(word in text for word in ["today", "aaj"]):
            return {"value": "today", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["tomorrow", "kal"]):
            return {"value": "tomorrow", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["day after tomorrow", "parso"]):
            return {"value": "day after tomorrow", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["immediately", "turant", "abhi", "now"]):
            return {"value": "immediately", "confidence": 0.9, "field": "availability_date", "method": "rule"}
        elif any(word in text for word in ["soon", "jaldi"]):
            return {"value": "soon", "confidence": 0.8, "field": "availability_date", "method": "rule"}
        
        # Look for date patterns (dd/mm/yyyy, dd-mm-yyyy)
        date_pattern = r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b'
        matches = re.findall(date_pattern, text)
        if matches:
            return {"value": matches[0], "confidence": 0.8, "field": "availability_date", "method": "regex"}
        
        return await self._extract_with_openai(text, "availability_date")
    
    async def _extract_shift(self, text: str) -> Dict[str, Any]:
        """Extract shift preference from text."""
        shift_keywords = {
            "morning": ["morning", "subah", "savera"],
            "afternoon": ["afternoon", "dopahar", "dupa"],
            "evening": ["evening", "shaam", "sham"],
            "night": ["night", "raat", "rat"],
            "flexible": ["flexible", "koi bhi", "any", "anytime"]
        }
        
        for shift, keywords in shift_keywords.items():
            if any(keyword in text for keyword in keywords):
                return {
                    "value": shift,
                    "confidence": 0.9,
                    "field": "preferred_shift",
                    "method": "rule"
                }
        
        return await self._extract_with_openai(text, "shift_preference")
    
    async def _extract_salary(self, text: str) -> Dict[str, Any]:
        """Extract salary amount from text with improved parsing."""
        text_lower = text.lower()
        
        # Enhanced patterns with better multiplier detection
        salary_patterns = [
            # Word + lakh patterns (highest priority)
            r'(?:fifteen|15|पंद्रह)\s*(?:lakh|lac|lakhs|लाख)',
            r'(?:twenty|20|बीस)\s*(?:lakh|lac|lakhs|लाख)',
            r'(?:twenty five|25|पच्चीस)\s*(?:lakh|lac|lakhs|लाख)',
            r'(?:thirty|30|तीस)\s*(?:lakh|lac|lakhs|लाख)',
            # Lakh patterns with numbers
            r'(\d+(?:\.\d+)?)\s*(?:lakh|lac|lakhs|लाख)',
            
            # Word + thousand patterns
            r'(?:fifteen|15|पंद्रह)\s*(?:thousand|hazaar|हजार|k)\b',
            r'(?:twenty|20|बीस)\s*(?:thousand|hazaar|हजार|k)\b',
            r'(?:twenty five|25|पच्चीस)\s*(?:thousand|hazaar|हजार|k)\b',
            r'(?:thirty|30|तीस)\s*(?:thousand|hazaar|हजार|k)\b',
            r'(?:forty|40|चालीस)\s*(?:thousand|hazaar|हजार|k)\b',
            r'(?:fifty|50|पचास)\s*(?:thousand|hazaar|हजार|k)\b',
            # Thousand patterns with numbers
            r'(\d+(?:\.\d+)?)\s*(?:thousand|hazaar|हजार|k)\b',
            
            # Rupees with explicit currency
            r'(?:rs\.?\s*|rupees?\s*|₹\s*)(\d+(?:,\d{3})*)',
            # Plain numbers (4-6 digits for salary range)
            r'\b(\d{4,6})\b'
        ]
        
        for i, pattern in enumerate(salary_patterns):
            match = re.search(pattern, text_lower)
            if match:
                try:
                    # Handle word-based patterns (lakh and thousand)
                    if i <= 3:  # Word + lakh patterns
                        if "fifteen" in match.group() or "15" in match.group() or "पंद्रह" in match.group():
                            amount = 15 * 100000
                        elif "twenty five" in match.group() or "25" in match.group() or "पच्चीस" in match.group():
                            amount = 25 * 100000
                        elif "twenty" in match.group() or "20" in match.group() or "बीस" in match.group():
                            amount = 20 * 100000
                        elif "thirty" in match.group() or "30" in match.group() or "तीस" in match.group():
                            amount = 30 * 100000
                        else:
                            continue
                        confidence = 0.95
                    elif i == 4:  # Numeric lakh patterns
                        amount_str = match.group(1).replace(',', '')
                        amount = float(amount_str) * 100000
                        confidence = 0.95
                    elif i <= 10:  # Word + thousand patterns
                        if "fifteen" in match.group() or "15" in match.group() or "पंद्रह" in match.group():
                            amount = 15000
                        elif "twenty five" in match.group() or "25" in match.group() or "पच्चीस" in match.group():
                            amount = 25000
                        elif "twenty" in match.group() or "20" in match.group() or "बीस" in match.group():
                            amount = 20000
                        elif "thirty" in match.group() or "30" in match.group() or "तीस" in match.group():
                            amount = 30000
                        elif "forty" in match.group() or "40" in match.group() or "चालीस" in match.group():
                            amount = 40000
                        elif "fifty" in match.group() or "50" in match.group() or "पचास" in match.group():
                            amount = 50000
                        else:
                            continue
                        confidence = 0.9
                    elif i == 11:  # Numeric thousand patterns  
                        amount_str = match.group(1).replace(',', '')
                        amount = float(amount_str) * 1000
                        confidence = 0.9
                    elif i == 12:  # Explicit currency
                        amount_str = match.group(1).replace(',', '')
                        amount = float(amount_str)
                        confidence = 0.9
                    else:  # Plain numbers
                        amount_str = match.group(1).replace(',', '')
                        amount = float(amount_str)
                        confidence = 0.8
                        # Heuristic: if number is very large, might be annual
                        if amount > 500000:
                            amount = amount / 12  # Convert annual to monthly
                    
                    # Validate reasonable salary range (1000 to 500000 monthly)
                    if 1000 <= amount <= 500000:
                        return {
                            "value": int(amount),
                            "confidence": confidence,
                            "field": "expected_salary",
                            "method": "regex"
                        }
                        
                except (ValueError, IndexError):
                    continue
        
        # Try LLM for complex salary expressions
        if settings.openai_api_key and openai is not None:
            try:
                prompt = (
                    "Extract monthly salary from user text. Return ONLY a JSON object.\n"
                    "Rules:\n"
                    "- Convert lakh to 100000 (e.g., 2.5 lakh = 250000)\n"  
                    "- Convert thousand to 1000 (e.g., 15 thousand = 15000)\n"
                    "- If yearly salary mentioned, divide by 12\n"
                    "- Return reasonable monthly amount between 5000-200000\n"
                    f"TEXT: '{text}'\n"
                    "Return: {\"monthly_salary\": <number>}"
                )
                resp = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0
                )
                content = resp.choices[0].message.content.strip()
                
                # Extract JSON
                import json as _json
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    data = _json.loads(content[start:end])
                    salary = int(data.get('monthly_salary', 0))
                    if 1000 <= salary <= 500000:
                        return {
                            "value": salary,
                            "confidence": 0.8,
                            "field": "expected_salary", 
                            "method": "llm"
                        }
            except Exception as e:
                logger.error(f"LLM salary extraction failed: {e}")
        
        # Fallback to simple number extraction
        numbers = re.findall(r'\d+', text)
        if numbers:
            amount = int(numbers[0])
            # Basic validation and conversion
            if amount < 1000:  # Too small, might need multiplier
                amount *= 1000
            elif amount > 500000:  # Too large, might be annual
                amount = amount // 12
            
            if 1000 <= amount <= 500000:
                return {
                    "value": amount,
                    "confidence": 0.6,
                    "field": "expected_salary",
                    "method": "fallback"
                }
        
        return await self._extract_with_openai(text, "salary")
    
    async def _extract_languages(self, text: str) -> Dict[str, Any]:
        """Extract language skills from text.
        Uses LLM primarily (negation + exclusivity aware), falls back to rule-based.
        Returns dict with keys: value (known enums), unknown (strings), confidence, field, method.
        """
        # Prefer LLM if configured
        if settings.openai_api_key and openai is not None:
            try:
                prompt = (
                    "You are extracting spoken languages from a short user utterance in Hinglish/English.\n"
                    "- Detect all languages the user says they CAN speak.\n"
                    "- Respect negations like 'nahi', 'nahin', 'not', 'cannot' (exclude those).\n"
                    "- Handle exclusivity like 'sirf/only X' meaning only X (exclude others).\n"
                    "- Handle phrases like 'hindi aur english' or 'hindi and english' (include both).\n"
                    "- For unclear responses like 'dono' or 'both', if context suggests hindi+english, include both.\n"
                    "- Return ONLY a JSON object with keys: known (list of enum names: hindi, english, marathi, bengali, tamil, telugu, gujarati, kannada, punjabi, malayalam, odia, assamese, urdu, nepali, kashmiri, sindhi, konkani, manipuri, maithili, santali, bodo, dogri, bhojpuri), unknown (list of other language names, title case). No extra text, no code fences.\n"
                    f"TEXT: {text}\n"
                )
                response = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.0
                )
                content = response.choices[0].message.content.strip()

                # Helper: strip code fences or extract JSON object substring
                def extract_json_blob(s: str) -> str:
                    import re as _re
                    m = _re.search(r"```(?:json)?\s*(\{[\s\S]*?\})\s*```", s, _re.IGNORECASE)
                    if m:
                        return m.group(1)
                    # Fallback: take substring between first '{' and last '}'
                    if '{' in s and '}' in s:
                        start = s.find('{')
                        end = s.rfind('}')
                        if start != -1 and end != -1 and end > start:
                            return s[start:end+1]
                    return s

                content = extract_json_blob(content)

                import json as _json
                data = _json.loads(content)

                known_raw = data.get("known", []) if isinstance(data, dict) else []
                unknown_raw = data.get("unknown", []) if isinstance(data, dict) else []

                # Sanitize helper
                import re as _re
                def clean_label(s: str) -> str:
                    s = (s or "").strip()
                    # Keep letters, spaces and hyphens
                    s = _re.sub(r"[^A-Za-z \-]", "", s)
                    return s.title().strip()

                stop = {"json", "other", "known", "unknown", "value", "field", "confidence", "method"}

                # Normalize known to enums and unknown to title case
                known_enums, unknown_norm = [], []
                for k in known_raw:
                    k = (k or "").strip().lower()
                    try:
                        LanguageSkill(k)  # validate
                        known_enums.append(k)
                    except Exception:
                        cl = clean_label(k)
                        if cl and cl.lower() not in stop:
                            unknown_norm.append(cl)
                for u in unknown_raw:
                    cl = clean_label(u)
                    if cl and cl.lower() not in stop:
                        unknown_norm.append(cl)

                # Deduplicate preserving order
                seen = set()
                unknown_norm = [x for x in unknown_norm if not (x.lower() in seen or seen.add(x.lower()))]

                return {"value": known_enums, "unknown": unknown_norm, "confidence": 0.8, "field": "languages", "method": "llm"}
            except Exception as e:
                logger.error(f"LLM language detection failed: {e}")
        
        # Enhanced rule-based fallback
        text_lower = text.lower()
        
        # Check for common combinations first
        if any(phrase in text_lower for phrase in ["dono", "both", "दोनो", "दोनों"]):
            # Default to hindi+english for "both"
            return {"value": ["hindi", "english"], "unknown": [], "confidence": 0.8, "field": "languages", "method": "rule"}
        
        if any(phrase in text_lower for phrase in ["hindi aur english", "hindi and english", "english aur hindi", "english and hindi"]):
            return {"value": ["hindi", "english"], "unknown": [], "confidence": 0.9, "field": "languages", "method": "rule"}
        
        # Enhanced language keywords with synonyms
        language_keywords = {
            "hindi": ["hindi", "हिंदी", "devanagari"],
            "english": ["english", "angrezi", "angrezee", "इंग्लिश"],
            "marathi": ["marathi", "मराठी", "maharashtrian"],
            "bengali": ["bengali", "bangla", "বাংলা", "bengalee"],
            "tamil": ["tamil", "தமிழ்", "tamizh"],
            "telugu": ["telugu", "తెలుగు", "telgu"],
            "gujarati": ["gujarati", "ગુજરાતી", "gujrati"],
            "kannada": ["kannada", "ಕನ್ನಡ", "kannad"],
            "punjabi": ["punjabi", "ਪੰਜਾਬੀ", "panjabi"],
            "malayalam": ["malayalam", "മലയാളം", "malayalee"],
            "odia": ["odia", "ଓଡ଼ିଆ", "oriya"],
            "assamese": ["assamese", "অসমীয়া", "asamiya"],
            "urdu": ["urdu", "اردو"],
            "nepali": ["nepali", "नेपाली"],
            "kashmiri": ["kashmiri", "کٲشُر"],
            "sindhi": ["sindhi", "سنڌي"],
            "konkani": ["konkani", "कोंकणी"],
            "manipuri": ["manipuri", "মেইতেইলোন"],
            "maithili": ["maithili", "मैथिली"],
            "santali": ["santali", "ᱥᱟᱱᱛᱟᱲᱤ"],
            "bodo": ["bodo", "बोडो"],
            "dogri": ["dogri", "डोगरी"],
            "bhojpuri": ["bhojpuri", "भोजपुरी"],
        }
        
        found_languages = []
        unknown_languages = []
        
        # Check for explicit negations
        negation_words = ["nahi", "nahin", "not", "cannot", "can't", "no", "नहीं", "sirf nahi", "bas nahi"]
        is_negated = any(neg in text_lower for neg in negation_words)
        
        # Check for exclusivity markers
        exclusive_words = ["sirf", "only", "keval", "केवल", "bas", "बस"]
        is_exclusive = any(exc in text_lower for exc in exclusive_words)
        
        for lang, keywords in language_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    # Check if this specific language is negated
                    lang_negated = any(f"{keyword} {neg}" in text_lower or f"{neg} {keyword}" in text_lower 
                                     for neg in negation_words)
                    
                    if not lang_negated and not (is_negated and not is_exclusive):
                        found_languages.append(lang)
                    break
        
        # Handle other potential languages not in enum
        other_lang_patterns = [
            "french", "spanish", "german", "arabic", "chinese", "japanese", "korean", 
            "portuguese", "russian", "italian", "dutch", "swedish", "norwegian"
        ]
        
        for lang in other_lang_patterns:
            if lang in text_lower:
                lang_negated = any(f"{lang} {neg}" in text_lower or f"{neg} {lang}" in text_lower 
                                 for neg in negation_words)
                if not lang_negated:
                    unknown_languages.append(lang.title())
        
        # Remove duplicates while preserving order
        found_languages = list(dict.fromkeys(found_languages))
        unknown_languages = list(dict.fromkeys(unknown_languages))
        
        if found_languages or unknown_languages:
            confidence = 0.8 if (found_languages and len(found_languages) <= 3) else 0.6
            return {"value": found_languages, "unknown": unknown_languages, "confidence": confidence, "field": "languages", "method": "rule"}
        
        return {"value": [], "unknown": [], "confidence": 0.0, "field": "languages", "method": "none"}
    
    async def _extract_boolean(self, text: str, field: str) -> Dict[str, Any]:
        """Extract yes/no response from text."""
        yes_keywords = ["yes", "haan", "han", "ji", "bilkul", "है", "h"]
        no_keywords = ["no", "nahi", "nahin", "na", "नहीं", "n"]
        
        if any(keyword in text.lower() for keyword in yes_keywords):
            return {"value": True, "confidence": 0.9, "field": field, "method": "rule"}
        elif any(keyword in text.lower() for keyword in no_keywords):
            return {"value": False, "confidence": 0.9, "field": field, "method": "rule"}
        
        return await self._extract_with_openai(text, field)
    
    async def _extract_experience(self, text: str) -> Dict[str, Any]:
        """Extract work experience in months with improved disambiguation."""
        text_lower = text.lower()
        
        # Check for "fresher" or "no experience" first
        fresher_keywords = ['fresher', 'fresh', 'naya', 'नया', 'no experience', 'koi experience nahi', 'experience nahi', 'kuch nahi']
        if any(word in text_lower for word in fresher_keywords):
            return {
                "value": 0,
                "confidence": 0.95,
                "field": "total_experience_months",
                "method": "rule"
            }
        
        # Enhanced patterns for better extraction
        year_patterns = [
            r'(\d+\.?\d*)\s*(?:saal|sal|years?|yrs?|yr)\b',
            r'(\d+\.?\d*)\s*(?:year|yer|yar|ear)\b'
        ]
        
        month_patterns = [
            r'(\d+\.?\d*)\s*(?:mahina|mahine|months?|mnths?|mths?|mons?)\b',
            r'(\d+\.?\d*)\s*(?:month|mnth|mth|mon)\b'
        ]
        
        total_months = 0
        found_explicit_unit = False
        
        # Look for explicit years
        for pattern in year_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                value = float(matches[0])
                total_months += int(round(value * 12))
                found_explicit_unit = True
                break
        
        # Look for explicit months (additive if we found years too)
        for pattern in month_patterns:
            matches = re.findall(pattern, text_lower)
            if matches:
                value = float(matches[0])
                total_months += int(round(value))
                found_explicit_unit = True
                break
        
        if found_explicit_unit and total_months >= 0:
            return {
                "value": total_months,
                "confidence": 0.9,
                "field": "total_experience_months",
                "method": "regex"
            }
        
        # Try LLM for better disambiguation of ambiguous cases
        if settings.openai_api_key and openai is not None:
            try:
                prompt = (
                    "Extract work experience from user text. Return ONLY a JSON object.\n"
                    "Rules:\n"
                    "- If they mention years (saal, years), multiply by 12\n"
                    "- If they mention months (mahina, months), use as-is\n"
                    "- For bare numbers < 5, assume YEARS and multiply by 12\n"
                    "- For bare numbers >= 5, assume MONTHS\n"
                    "- If fresher/no experience, return 0\n"
                    f"TEXT: '{text}'\n"
                    "Return: {\"months\": <number>}"
                )
                resp = await asyncio.to_thread(
                    openai.chat.completions.create,
                    model="gpt-4o",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=50,
                    temperature=0.0
                )
                content = resp.choices[0].message.content.strip()
                
                # Extract JSON
                import json as _json
                if '{' in content and '}' in content:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    data = _json.loads(content[start:end])
                    months = int(data.get('months', 0))
                    if months >= 0:
                        return {
                            "value": months,
                            "confidence": 0.8,
                            "field": "total_experience_months",
                            "method": "llm"
                        }
            except Exception as e:
                logger.error(f"LLM experience extraction failed: {e}")
        
        # Fallback for bare numbers with improved heuristics
        bare_num = re.search(r'\b(\d{1,2}(?:\.\d+)?)\b', text_lower)
        if bare_num:
            val = float(bare_num.group(1))
            # Better heuristic: numbers < 5 likely years, >= 5 likely months
            if val < 5:
                months = int(round(val * 12))  # Assume years
                confidence = 0.7
            else:
                months = int(round(val))  # Assume months
                confidence = 0.6
            
            return {
                "value": months,
                "confidence": confidence,
                "field": "total_experience_months",
                "method": "heuristic"
            }
        
        # Fallback to OpenAI for complex cases
        return await self._extract_with_openai(text, "experience")
    
    async def _extract_with_openai(self, text: str, field: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
        """Use OpenAI for complex entity extraction."""
        if not settings.openai_api_key:
            return {"value": None, "confidence": 0.0, "field": field, "method": "none"}
        
        try:
            prompt = self._create_extraction_prompt(text, field, context)
            
            response = await asyncio.to_thread(
                openai.chat.completions.create,
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": (
                        "Extract the requested entity ONLY. Return the minimal value with no extra words, no labels, no punctuation, no quotes. "
                        "Examples of correct outputs: 2 months, Chennai, 20000, today, night, yes."
                    )},
                    {"role": "user", "content": prompt},
                ],
                max_tokens=150,
                temperature=0.1
            )
            
            result_text = response.choices[0].message.content.strip()
            
            # Parse the response
            return self._parse_openai_response(result_text, field)
            
        except Exception as e:
            logger.error(f"OpenAI extraction failed: {e}")
            return {"value": None, "confidence": 0.0, "field": field, "method": "failed"}
 
    def _create_extraction_prompt(self, text: str, field: str, context: Dict[str, Any] = None) -> str:
        """Create prompt for OpenAI entity extraction."""
        header = (
            "Task: Extract ONLY the entity, strictly and minimally. No extra words, labels, punctuation, or quotes.\n"
            "Good outputs: 2 months, Chennai, 20000, today, night, yes\n"
        )
        prompts = {
            "pincode": (
                f"{header}From: '{text}'\n"
                "Output a 6-digit pincode. If not present, output a single locality/city token (e.g., Chennai)."
            ),
            "salary": (
                f"{header}From: '{text}'\n"
                "Output monthly salary in INR as digits only (commas allowed), e.g., 20000 or 20,000."
            ),
            "availability_date": (
                f"{header}From: '{text}'\n"
                "Output one of: today, tomorrow, soon, or a date in dd/mm/yyyy or dd-mm-yyyy."
            ),
            "shift_preference": (
                f"{header}From: '{text}'\n"
                "Output exactly one: morning, afternoon, evening, night, or flexible."
            ),
            "languages": (
                # Languages use a separate JSON format elsewhere; keep instruction minimal here for fallback
                f"From: '{text}'\nReturn ONLY a comma-separated list of language names (e.g., Hindi, English)."
            ),
            "experience": (
                f"{header}From: '{text}'\n"
                "Output total work experience as 'N months' with N as an integer (e.g., 2 months)."
            ),
            "has_two_wheeler": (
                f"{header}From: '{text}'\n"
                "Output exactly yes or no."
            ),
        }
        
        return prompts.get(field, f"{header}Extract relevant information for {field} from: '{text}'")
 
    def _parse_openai_response(self, response: str, field: str) -> Dict[str, Any]:
        """Parse OpenAI response into structured format."""
        response = response.strip().lower()
        
        if response == "none" or not response:
            return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        # Field-specific parsing
        if field == "expected_salary":
            try:
                import re as _re
                digits = _re.sub(r"[^0-9]", "", response)
                value = int(digits) if digits else None
                if value is not None:
                    return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
            except Exception:
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        elif field == "has_two_wheeler":
            value = response in ["yes", "true", "1", "haan"]
            return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
        
        elif field == "total_experience_months":
            try:
                import re as _re
                digits = _re.sub(r"[^0-9]", "", response)
                value = int(digits) if digits else None
                if value is not None:
                    return {"value": value, "confidence": 0.7, "field": field, "method": "openai"}
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
            except Exception:
                return {"value": None, "confidence": 0.0, "field": field, "method": "openai"}
        
        else:
            return {"value": response, "confidence": 0.7, "field": field, "method": "openai"} 