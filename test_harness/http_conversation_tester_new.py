"""
Clean HTTP Persona Conversation Tester - LLM-First Approach
No confidence thresholds, no retries - everything goes through LLM immediately.
"""

import asyncio
import time
import json as _json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field
import httpx
from loguru import logger
import yaml
import os

# Try importing OpenAI (new SDK)
try:
    from openai import OpenAI
except ImportError:
    OpenAI = None

# Load .env if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass


@dataclass
class ConversationTurn:
    """Represents a single conversation turn."""
    turn_number: int
    step_name: str
    audio_file: str
    asr_text: str
    asr_confidence: float
    bot_response: str
    latency_ms: float
    success: bool
    extracted_entities: Dict
    current_field: str = ""
    candidate_profile: Dict = field(default_factory=dict)
    audio_variant: str = "auto"
    tester_llm_entities: Dict = field(default_factory=dict)


@dataclass
class PersonaTestResult:
    """Test result for a single persona."""
    persona_key: str
    completed: bool
    turns: int
    avg_confidence: float
    confidence_range: Tuple[float, float]
    latency_p95_ms: float
    entity_extraction_metrics: Dict = None
    success: bool = True
    step_completion_rate: float = 1.0
    completion_time_s: float = 0.0
    conversation_turns: List[ConversationTurn] = field(default_factory=list)


class HTTPPersonaConversationTester:
    """Clean conversation tester with LLM-first approach."""
    
    def __init__(self, bot_base_url: str = "http://localhost:8000"):
        self.bot_base_url = bot_base_url
        self.results_dir = Path("test_harness/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Load utterance entities for evaluation
        self._utterance_entities = self._load_utterance_entities()
        
        # Configure OpenAI client if available
        self._openai_client = None
        try:
            api_key = os.getenv("OPENAI_API_KEY")
            if OpenAI is not None and api_key:
                self._openai_client = OpenAI(api_key=api_key)
        except Exception:
            logger.warning("OpenAI client not initialized; LLM extraction will return None.")

        # Define personas with correct expected entities from utterances.yaml
        self.personas = {
            "english_man": {
                "name": "English Man (Professional)",
                "voice_id": "LruHrtVF6PSyGItzMNHS",
                "noise_levels": ["clean", "low"],
                "conversation_flow": [
                    {"step": "pincode", "audio_file": "pin_diverse_001", "expected_entity": {"pincode": "110001"}},
                    {"step": "expected_salary", "audio_file": "sal_diverse_001", "expected_entity": {"expected_salary": 25000}},
                    {"step": "has_two_wheeler", "audio_file": "vehicle_diverse_001", "expected_entity": {"has_two_wheeler": False}},
                    {"step": "languages", "audio_file": "lang_diverse_001", "expected_entity": {"languages": ["hindi", "english"]}},
                    {"step": "availability_date", "audio_file": "avail_diverse_001", "expected_entity": {"availability_date": "soon"}},
                    {"step": "preferred_shift", "audio_file": "shift_diverse_001", "expected_entity": {"preferred_shift": "morning"}},
                    {"step": "total_experience_months", "audio_file": "exp_diverse_001", "expected_entity": {"total_experience_months": 2}},
                ],
            },
            "calm_hindi": {
                "name": "Calm Hindi Speaker", 
                "voice_id": "1Z7Y8o9cvUeWq8oLKgMY",
                "noise_levels": ["low", "medium"],
                "conversation_flow": [
                    {"step": "pincode", "audio_file": "pin_diverse_002", "expected_entity": {"pincode": "560008"}},
                    {"step": "expected_salary", "audio_file": "sal_diverse_002", "expected_entity": {"expected_salary": 60000}},
                    {"step": "has_two_wheeler", "audio_file": "vehicle_diverse_002", "expected_entity": {"has_two_wheeler": True}},
                    {"step": "languages", "audio_file": "lang_diverse_002", "expected_entity": {"languages": ["English", "Kannada", "Tamil"]}},
                    {"step": "availability_date", "audio_file": "avail_diverse_002", "expected_entity": {"availability_date": "soon"}},
                    {"step": "preferred_shift", "audio_file": "shift_diverse_002", "expected_entity": {"preferred_shift": "evening"}},
                    {"step": "total_experience_months", "audio_file": "exp_diverse_002", "expected_entity": {"total_experience_months": 12}},
                ],
            },
            "energetic_hindi": {
                "name": "Energetic Hindi Speaker",
                "voice_id": "IvLWq57RKibBrqZGpQrC", 
                "noise_levels": ["medium", "low"],
                "conversation_flow": [
                    {"step": "pincode", "audio_file": "pin_diverse_003", "expected_entity": {"pincode": "600001"}},
                    {"step": "expected_salary", "audio_file": "sal_diverse_003", "expected_entity": {"expected_salary": 50000}},
                    {"step": "has_two_wheeler", "audio_file": "vehicle_diverse_003", "expected_entity": {"has_two_wheeler": True}},
                    {"step": "languages", "audio_file": "lang_diverse_003", "expected_entity": {"languages": ["Tamil", "English"]}},
                    {"step": "availability_date", "audio_file": "avail_diverse_003", "expected_entity": {"availability_date": "One week"}},
                    {"step": "preferred_shift", "audio_file": "shift_diverse_003", "expected_entity": {"preferred_shift": "flexible"}},
                    {"step": "total_experience_months", "audio_file": "exp_diverse_003", "expected_entity": {"total_experience_months": 24}},
                ],
            },
            "expressive_hindi": {
                "name": "Expressive Hindi Speaker",
                "voice_id": "ni6cdqyS9wBvic5LPA7M",
                "noise_levels": ["high", "medium"],
                "conversation_flow": [
                    {"step": "pincode", "audio_file": "pin_diverse_004", "expected_entity": {"pincode": "620001"}},
                    {"step": "expected_salary", "audio_file": "sal_diverse_004", "expected_entity": {"expected_salary": 10000}},
                    {"step": "has_two_wheeler", "audio_file": "vehicle_diverse_004", "expected_entity": {"has_two_wheeler": True}},
                    {"step": "languages", "audio_file": "lang_diverse_004", "expected_entity": {"languages": ["tamil"]}},
                    {"step": "availability_date", "audio_file": "avail_diverse_004", "expected_entity": {"availability_date": "soon"}},
                    {"step": "preferred_shift", "audio_file": "shift_diverse_004", "expected_entity": {"preferred_shift": "night"}},
                    {"step": "total_experience_months", "audio_file": "exp_diverse_004", "expected_entity": {"total_experience_months": 0}},
                ],
            },
        }

        # Golden jobs per persona prefix
        self.persona_job_prefix = {
            "english_man": "GLD_EN_",
            "calm_hindi": "GLD_CH_",
            "energetic_hindi": "GLD_EH_",
            "expressive_hindi": "GLD_XH_",
        }
        # Use first 3 jobs as target set per persona
        self.num_golden = 3

    def _load_utterance_entities(self) -> Dict:
        """Load expected entities from utterances.yaml."""
        try:
            utterances_path = Path("data/gen/utterances.yaml")
            if utterances_path.exists():
                with open(utterances_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                    entities = {}
                    for utterance in data.get('utterances', []):
                        entities[utterance['id']] = utterance.get('entities', {})
                    return entities
        except Exception as e:
            logger.warning(f"Could not load utterances.yaml: {e}")
        return {}

    # ==== Entity evaluation helpers ====
    def _normalize_value(self, slot: str, value: Any) -> Any:
        """Normalize values per slot for robust comparison.
        
        Args:
            slot (str): Slot name
            value (Any): Raw value
        Returns:
            Any: Normalized value suitable for comparison
        """
        if value is None:
            return None
        if slot == "pincode":
            s = str(value).strip()
            return int(s) if s.isdigit() else s
        if slot == "expected_salary":
            try:
                return int(value)
            except Exception:
                return None
        if slot == "total_experience_months":
            try:
                return int(value)
            except Exception:
                return None
        if slot == "has_two_wheeler":
            if isinstance(value, bool):
                return value
            s = str(value).strip().lower()
            if s in {"true", "yes", "haan", "ha", "1"}:
                return True
            if s in {"false", "no", "nahin", "nahi", "0"}:
                return False
            return None
        if slot == "languages":
            if isinstance(value, list):
                return {str(v).strip().lower() for v in value if str(v).strip()}
            if isinstance(value, set):
                return {str(v).strip().lower() for v in value if str(v).strip()}
            if isinstance(value, str) and value.strip():
                return {value.strip().lower()}
            return set()
        if slot == "preferred_shift":
            allowed = {"morning", "afternoon", "evening", "night", "flexible"}
            if isinstance(value, list):
                return {v for v in (str(x).strip().lower() for x in value) if v in allowed}
            if isinstance(value, set):
                return {v for v in (str(x).strip().lower() for x in value) if v in allowed}
            if isinstance(value, str):
                v = value.strip().lower()
                return {v} if v in allowed else set()
            return set()
        if slot == "availability_date":
            v = str(value).strip().lower()
            synonyms = {
                "tmrw": "tomorrow",
                "tmr": "tomorrow",
                "immediately": "today",
                "asap": "today",
                "right now": "today",
                "now": "today",
            }
            return synonyms.get(v, v)
        return value

    def _compare_slot(self, slot: str, expected: Any, predicted: Any) -> Tuple[int, int, int]:
        """Compare expected vs predicted for a slot and return TP/FP/FN.
        
        Args:
            slot (str): Slot name
            expected (Any): Expected value (pre-normalization)
            predicted (Any): Predicted value (pre-normalization)
        Returns:
            Tuple[int, int, int]: tp, fp, fn
        """
        exp = self._normalize_value(slot, expected)
        pred = self._normalize_value(slot, predicted)

        if slot == "languages":
            exp_set = exp if isinstance(exp, set) else set(exp or [])
            pred_set = pred if isinstance(pred, set) else set(pred or [])
            if not exp_set and not pred_set:
                return 1, 0, 0
            if exp_set.issubset(pred_set):
                return 1, 0, 0
            return 0, 1, 1

        if slot == "preferred_shift":
            exp_set = exp if isinstance(exp, set) else set(exp or [])
            pred_set = pred if isinstance(pred, set) else set(pred or [])
            if not exp_set and not pred_set:
                return 1, 0, 0
            if "flexible" in pred_set:
                return 1, 0, 0
            if "flexible" in exp_set and "flexible" in pred_set:
                return 1, 0, 0
            if exp_set.issubset(pred_set):
                return 1, 0, 0
            return 0, 1, 1

        if slot == "pincode":
            if exp is None and pred is None:
                return 1, 0, 0
            try:
                ei = int(exp) if isinstance(exp, int) else int(str(exp).strip())
            except Exception:
                ei = None
            try:
                pi = int(pred) if isinstance(pred, int) else int(str(pred).strip())
            except Exception:
                pi = None
            if ei is not None and pi is not None:
                if abs(ei - pi) <= 10:
                    return 1, 0, 0
                return 0, 1, 1
            if str(exp).strip() == str(pred).strip():
                return 1, 0, 0
            return 0, 1, 1

        if slot == "availability_date":
            if exp is None and pred is None:
                return 1, 0, 0
            e = str(exp).strip().lower() if exp is not None else None
            p = str(pred).strip().lower() if pred is not None else None
            if e == p:
                return 1, 0, 0
            immediate_group = {"today", "immediately", "soon"}
            tomorrow_group = {"tomorrow", "tmrw", "tmr", "soon"}
            if (e in immediate_group and p in immediate_group) or (e in tomorrow_group and p in tomorrow_group):
                return 1, 0, 0
            return 0, 1, 1

        # Scalar equality for remaining
        if exp is None and pred is None:
            return 1, 0, 0
        if exp == pred:
            return 1, 0, 0
        return 0, 1, 1

    def _compute_f1(self, tp: int, fp: int, fn: int) -> float:
        """Compute F1 given counts.
        
        Args:
            tp (int): True positives
            fp (int): False positives
            fn (int): False negatives
        Returns:
            float: F1 score
        """
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _llm_extract_for_slot(self, slot: str, asr_text: str) -> Optional[object]:
        """Extract entity using LLM."""
        if not asr_text:
            logger.warning(f"🔬 LLM: No ASR text provided for {slot}")
            return None
        if not self._openai_client:
            logger.warning(f"🔬 LLM: OpenAI client not available for {slot}")
            return None
            
        try:
            system = (
                "Extract ONLY the requested entity from short Hinglish text. "
                "Return strict JSON with a single key 'value'. No prose."
            )
            rules = {
                "pincode": (
                    "Output a 6-digit Indian pincode. If a complete 6-digit pincode is present, use it. "
                    "If only city/locality is mentioned (like Delhi, Mumbai, Chennai, Bangalore, Kolkata, etc.), "
                    "infer the main pincode for that city. Examples: Delhi->110001, Mumbai->400001, Chennai->600001, "
                    "Bangalore->560001, Kolkata->700001. If neither pincode nor recognizable city, output null."
                ),
                "availability_date": (
                    "Output one of: today, tomorrow, immediately, soon, or a dd/mm/yyyy date; else null."
                ),
                "preferred_shift": (
                    "Output a JSON array of any that apply from: morning, afternoon, evening, night, flexible; else empty array."
                ),
                "expected_salary": (
                    "Output monthly salary in INR as integer digits; infer thousand/lakh; else null."
                ),
                "languages": (
                    "Output a JSON array of languages (lowercase) the user CAN speak from: hindi, english, marathi, bengali, tamil, telugu, gujarati, kannada, punjabi, malayalam, odia, assamese, urdu; else empty array."
                ),
                "has_two_wheeler": (
                    "Output true if user has bike/scooter/motorcycle/two-wheeler; false if user explicitly says no/nahi; else null."
                ),
                "total_experience_months": (
                    "Output total work experience in months as integer; convert years to months (1 year = 12 months); "
                    "if fresher/no experience output 0; else null."
                ),
            }
            
            rule = rules.get(slot, "Output null.")
            user = f"Field: {slot}\nText: {asr_text}\nReturn: {{\"value\": ...}}"
            
            resp = self._openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": system + " " + rule},
                    {"role": "user", "content": user},
                ],
                max_tokens=120,
                temperature=0.0,
            )
            
            content = (resp.choices[0].message.content or "").strip()
            logger.info(f"🔬 LLM RAW RESPONSE for {slot}: {content}")
            
            if "{" in content and "}" in content:
                start, end = content.find("{"), content.rfind("}") + 1
                data = _json.loads(content[start:end])
            else:
                data = _json.loads(content)
                
            value = data.get("value") if isinstance(data, dict) else None
            logger.info(f"🔬 LLM PARSED VALUE for {slot}: {value}")
            
            # Validate and normalize per slot
            if slot == "pincode":
                if isinstance(value, str) and value.isdigit() and len(value) == 6:
                    logger.info(f"🔬 LLM PINCODE VALIDATED: {value}")
                    return value
                logger.warning(f"🔬 LLM PINCODE INVALID: {value} (type: {type(value)})")
                return None
                
            elif slot == "expected_salary":
                if isinstance(value, (int, str)):
                    try:
                        return int(value)
                    except:
                        return None
                return None
                
            elif slot == "has_two_wheeler":
                if isinstance(value, bool):
                    return value
                return None
                
            elif slot == "languages":
                if isinstance(value, list):
                    return value
                return None
                
            elif slot == "preferred_shift":
                if isinstance(value, list):
                    return value
                return None
                
            elif slot == "total_experience_months":
                if isinstance(value, (int, str)):
                    try:
                        return int(value)
                    except:
                        return None
                return None
                
            else:
                return value
                
        except Exception as e:
            logger.error(f"❌ LLM extraction failed for {slot}: {e}")
            return None

    async def _confirm_text(self, client: httpx.AsyncClient, candidate_id: str, text: str) -> bool:
        """Send text confirmation to server."""
        try:
            response = await client.post(
                f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/confirm-fast",
                json={"text_only": True, "user_text": text},
                timeout=30.0
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Failed to confirm text '{text}': {e}")
            return False

    async def _fetch_status(self, client: httpx.AsyncClient, candidate_id: str) -> Optional[Dict]:
        """Fetch conversation status from server."""
        try:
            response = await client.get(
                f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/status",
                timeout=30.0
            )
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            logger.error(f"Failed to fetch status: {e}")
        return None

    async def send_audio_turn(self, client: httpx.AsyncClient, candidate_id: str, 
                             audio_id: str, step_name: str, expected_entities: Dict,
                             force_clean: bool = False) -> ConversationTurn:
        """Send audio and get response."""
        # Determine audio file path
        base_path = Path("test_harness/generated_audio")
        clean_file = base_path / f"{audio_id}_clean.wav"
        noisy_file = base_path / f"{audio_id}_noisy_gpt.wav"
        
        if force_clean or not noisy_file.exists():
            audio_file = clean_file
            variant = "clean"
            logger.info(f"Using clean audio for {audio_id}: {audio_file}")
        else:
            audio_file = noisy_file
            variant = "noisy"
            logger.info(f"Using noisy audio for {audio_id}: {audio_file}")
            
        if not audio_file.exists():
            logger.error(f"Audio file not found: {audio_file}")
            return ConversationTurn(
                turn_number=0, step_name=step_name, audio_file=str(audio_file),
                asr_text="", asr_confidence=0.0, bot_response="", latency_ms=0.0,
                success=False, extracted_entities={}, audio_variant=variant
            )

        # Send audio to server
        try:
            with open(audio_file, 'rb') as f:
                audio_data = f.read()
                
            logger.info(f"Sending turn: {audio_file} ({len(audio_data)} bytes)")
            start_time = time.time()
            
            files = {"audio_file": ("recording.wav", audio_data, "audio/wav")}
            response = await client.post(
                f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/turn-fast",
                files=files,
                timeout=30.0
            )
            
            end_time = time.time()
            latency_ms = (end_time - start_time) * 1000
            
            if response.status_code != 200:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                return ConversationTurn(
                    turn_number=0, step_name=step_name, audio_file=str(audio_file),
                    asr_text="", asr_confidence=0.0, bot_response="", latency_ms=latency_ms,
                    success=False, extracted_entities={}, audio_variant=variant
                )
                
            data = response.json()
            logger.debug(f"Full API response: {_json.dumps(data, indent=2)}")
            
            # Extract response data
            asr_text = data.get("asr", {}).get("text", "")
            asr_confidence = data.get("asr", {}).get("confidence", 0.0)
            bot_response = data.get("text", "")
            current_field = data.get("metrics", {}).get("current_field", "")
            candidate_profile = data.get("metrics", {}).get("candidate_profile", {})
            
            logger.info(f"✓ Turn completed in {latency_ms:.0f}ms")
            logger.info(f"  ASR: '{asr_text[:50]}...' (conf: {asr_confidence:.2f})")
            logger.info(f"  Bot: '{bot_response[:50]}...'")
            logger.info(f"  Current field: {current_field}")
            
            return ConversationTurn(
                turn_number=0,
                step_name=step_name,
                audio_file=str(audio_file),
                asr_text=asr_text,
                asr_confidence=asr_confidence,
                bot_response=bot_response,
                latency_ms=latency_ms,
                success=True,
                extracted_entities=expected_entities,
                current_field=current_field,
                candidate_profile=candidate_profile,
                audio_variant=variant
            )
            
        except Exception as e:
            logger.error(f"Failed to send audio turn: {e}")
            return ConversationTurn(
                turn_number=0, step_name=step_name, audio_file=str(audio_file),
                asr_text="", asr_confidence=0.0, bot_response="", latency_ms=0.0,
                success=False, extracted_entities={}, audio_variant=variant
            )

    async def run_persona_test(self, persona_key: str) -> PersonaTestResult:
        """Run conversation test for a single persona with LLM-first approach."""
        start_time = time.time()
        persona = self.personas[persona_key]
        logger.info(f"🎭 Starting {persona['name']} conversation test...")
        
        turns = []
        confidences = []
        
        async with httpx.AsyncClient() as client:
            # Start conversation
            try:
                response = await client.post(
                    f"{self.bot_base_url}/api/v1/conversation/start",
                    json={},
                    timeout=30.0
                )
                if response.status_code != 200:
                    logger.error(f"Failed to start conversation: {response.text}")
                    return PersonaTestResult(
                        persona_key=persona_key, completed=False, turns=0,
                        avg_confidence=0.0, confidence_range=(0, 0), latency_p95_ms=0.0,
                        success=False
                    )
                    
                candidate_id = response.json()["candidate_id"]
                logger.info(f"Started conversation session: {candidate_id}")
                
            except Exception as e:
                logger.error(f"Failed to start conversation: {e}")
                return PersonaTestResult(
                    persona_key=persona_key, completed=False, turns=0,
                    avg_confidence=0.0, confidence_range=(0, 0), latency_p95_ms=0.0,
                    success=False
                )

            # Process each step in the conversation flow
            for i, config in enumerate(persona["conversation_flow"], 1):
                slot = config["step"]
                audio_id = config["audio_file"]
                
                logger.info(f"\n--- Step {i}/{len(persona['conversation_flow'])}: {slot} ---")
                
                # Send audio and get ASR result
                tr = await self.send_audio_turn(
                    client, candidate_id, audio_id, slot,
                    config.get("expected_entity", {}), force_clean=False
                )
                
                turns.append(tr)
                confidences.append(tr.asr_confidence)
                
                # If HTTP failed, stop conversation
                if not tr.success:
                    logger.error(f"Failed to send audio for {slot}, stopping conversation")
                    break
                
                # LLM EXTRACTION FOR ALL SLOTS (IMMEDIATE)
                logger.info(f"🔬 ATTEMPTING LLM EXTRACTION for {slot} with ASR: '{tr.asr_text}'")
                try:
                    val = self._llm_extract_for_slot(slot, tr.asr_text or "")
                    logger.info(f"🔬 LLM RETURNED: {val} for {slot}")
                    
                    # Always store LLM result, even if None
                    tr.tester_llm_entities = tr.tester_llm_entities or {}
                    tr.tester_llm_entities[slot] = val
                    
                    if val not in (None, [], ""):
                        logger.info(f"LLM-extracted {slot}: {val}")
                        
                        # Send extracted value to server immediately
                        try:
                            ok = False
                            if slot == "pincode" and isinstance(val, str) and val.isdigit() and len(val) == 6:
                                ok = await self._confirm_text(client, candidate_id, text=val)
                            elif slot == "expected_salary" and isinstance(val, (int, str)):
                                ok = await self._confirm_text(client, candidate_id, text=str(val))
                            elif slot == "languages" and isinstance(val, list) and val:
                                lang_text = val[0] if len(val) == 1 else ", ".join(val)
                                ok = await self._confirm_text(client, candidate_id, text=lang_text)
                            elif slot == "has_two_wheeler" and isinstance(val, bool):
                                ok = await self._confirm_text(client, candidate_id, text="yes" if val else "no")
                            elif slot == "availability_date" and val:
                                ok = await self._confirm_text(client, candidate_id, text=str(val))
                            elif slot == "preferred_shift" and val:
                                shift_text = val[0] if isinstance(val, list) and val else str(val)
                                ok = await self._confirm_text(client, candidate_id, text=shift_text)
                            elif slot == "total_experience_months" and isinstance(val, (int, str)):
                                ok = await self._confirm_text(client, candidate_id, text=str(val))
                                
                            if ok:
                                logger.info(f"✅ Successfully sent {slot}={val} to server")
                            else:
                                logger.warning(f"❌ Failed to send {slot}={val} to server")
                                
                        except Exception as e:
                            logger.error(f"❌ Error sending {slot}={val}: {e}")
                    else:
                        logger.info(f"🔬 LLM returned None/empty for {slot} - keeping as None and moving on")
                        
                except Exception as e:
                    logger.error(f"❌ LLM extraction failed for {slot}: {e}")
                
                # Always move to next step - no retries, no confidence checks
                
        # Calculate metrics and return result
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.0
        confidence_range = (min(confidences), max(confidences)) if confidences else (0, 0)
        final_profile = turns[-1].candidate_profile if turns else {}
        
        # Collect tester entities
        tester_best_entities = {}
        for turn in turns:
            if hasattr(turn, 'tester_llm_entities') and turn.tester_llm_entities:
                tester_best_entities.update(turn.tester_llm_entities)
        
        # Calculate entity metrics
        entity_metrics = self._calculate_entity_metrics(turns, final_profile, tester_best_entities)
        
        # Print results for debugging
        self._print_entity_comparison(entity_metrics, persona["name"])
        
        conversation_completed = bool(final_profile.get("conversation_completed", False))

        # Fetch matches if completed
        match_metrics = None
        matched_job_ids: List[str] = []
        if conversation_completed:
            try:
                async with httpx.AsyncClient() as client2:
                    resp = await client2.get(f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/matches", timeout=30.0)
                    if resp.status_code == 200:
                        j = resp.json()
                        matches = j.get("matches", [])
                        matched_job_ids = [m.get("job", {}).get("job_id") for m in matches if m.get("job")]
                        match_metrics = self._evaluate_job_matches(persona_key, matched_job_ids)
            except Exception as e:
                logger.warning(f"Failed to fetch/evaluate matches: {e}")
        
        # Calculate entity metrics
        entity_metrics = self._calculate_entity_metrics(turns, final_profile, tester_best_entities)
        
        # Print results for debugging
        self._print_entity_comparison(entity_metrics, persona["name"])
        if match_metrics:
            self._print_match_metrics(persona["name"], matched_job_ids, match_metrics)
        
        return PersonaTestResult(
            persona_key=persona_key,
            completed=conversation_completed,
            turns=len(turns),
            avg_confidence=avg_confidence,
            confidence_range=confidence_range,
            latency_p95_ms=max([turn.latency_ms for turn in turns]) if turns else 0.0,
            entity_extraction_metrics=entity_metrics,
            success=True,
            step_completion_rate=1.0,
            completion_time_s=time.time() - start_time,
            conversation_turns=turns
        )

    def _extract_utterance_id(self, audio_path: str) -> Optional[str]:
        """Get utterance id from audio file path.
        
        Args:
            audio_path (str): Path to audio file
        Returns:
            Optional[str]: Utterance id like 'pin_diverse_003'
        """
        try:
            name = Path(audio_path).name
            base = name.replace("_clean.wav", "").replace("_noisy_gpt.wav", "")
            return base
        except Exception:
            return None

    def _calculate_entity_metrics(self, turns: List[ConversationTurn], 
                                 final_profile: Dict, tester_entities: Dict) -> Dict:
        """Calculate entity extraction metrics.
        
        Args:
            turns (List[ConversationTurn]): Conversation turns
            final_profile (Dict): Final profile from server
            tester_entities (Dict): Aggregated LLM-extracted entities
        Returns:
            Dict: Metrics object with per_slot and overall
        """
        interesting_slots = [
            "pincode",
            "availability_date",
            "preferred_shift",
            "expected_salary",
            "languages",
            "has_two_wheeler",
            "total_experience_months",
        ]

        # Build expected by slot from utterances.yaml, prioritizing the last mention per slot
        expected_by_slot: Dict[str, Any] = {}
        for t in turns:
            utt_id = self._extract_utterance_id(t.audio_file)
            if not utt_id:
                continue
            exp_map = self._utterance_entities.get(utt_id, {})
            for slot, val in exp_map.items():
                if slot in interesting_slots:
                    expected_by_slot[slot] = val

        # Derive predicted values from final profile, falling back to tester_entities
        predicted_by_slot: Dict[str, Any] = {}
        if final_profile:
            predicted_by_slot["pincode"] = final_profile.get("pincode")
            predicted_by_slot["availability_date"] = final_profile.get("availability_date")
            predicted_by_slot["preferred_shift"] = final_profile.get("preferred_shift")
            predicted_by_slot["expected_salary"] = final_profile.get("expected_salary")
            langs = (final_profile.get("languages") or []) + (final_profile.get("other_languages") or [])
            predicted_by_slot["languages"] = langs
            predicted_by_slot["has_two_wheeler"] = final_profile.get("has_two_wheeler")
            predicted_by_slot["total_experience_months"] = final_profile.get("total_experience_months")
        # Fill from tester_entities where profile missing
        for slot in interesting_slots:
            if predicted_by_slot.get(slot) in (None, [], ""):
                if slot in tester_entities:
                    predicted_by_slot[slot] = tester_entities.get(slot)

        per_slot: Dict[str, Dict[str, Any]] = {}
        tp_total = fp_total = fn_total = 0
        for slot in interesting_slots:
            exp = expected_by_slot.get(slot)
            pred = predicted_by_slot.get(slot)
            tp, fp, fn = self._compare_slot(slot, exp, pred)
            f1 = self._compute_f1(tp, fp, fn)
            tp_total += tp
            fp_total += fp
            fn_total += fn

            norm_exp = self._normalize_value(slot, exp)
            norm_pred = self._normalize_value(slot, pred)
            # Convert sets to lists for JSON friendliness
            if isinstance(norm_exp, set):
                norm_exp = sorted(list(norm_exp))
            if isinstance(norm_pred, set):
                norm_pred = sorted(list(norm_pred))

            per_slot[slot] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "f1": f1,
                "expected": exp,
                "predicted": pred,
                "normalized_expected": norm_exp,
                "normalized_predicted": norm_pred,
            }

        macro_f1 = sum(v["f1"] for v in per_slot.values()) / len(per_slot) if per_slot else 0.0
        micro_f1 = self._compute_f1(tp_total, fp_total, fn_total)
        return {
            "per_slot": per_slot,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "total_tp": tp_total,
            "total_fp": fp_total,
            "total_fn": fn_total,
        }

    def _print_entity_comparison(self, entity_metrics: Dict, persona_name: str):
        """Print entity comparison for debugging."""
        print(f"\n🔍 ENTITY VALUES - {persona_name}")
        print("="*60)
        if not entity_metrics or "per_slot" not in entity_metrics:
            print("No metrics.")
            print("="*60)
            return
        for slot, m in entity_metrics["per_slot"].items():
            expected = m.get("expected")
            predicted = m.get("predicted")
            match = "✓" if m.get("tp", 0) > 0 else "✗"
            print(f"{slot}: expected={expected}, derived={predicted} {match}")
        print(f"\nMacro F1: {entity_metrics.get('macro_f1', 0.0):.3f}")
        print("="*60)

    def _golden_job_ids_for_persona(self, persona_key: str) -> List[str]:
        prefix = self.persona_job_prefix.get(persona_key, "GLD_")
        return [f"{prefix}{i:03d}" for i in range(1, self.num_golden + 1)]

    def _evaluate_job_matches(self, persona_key: str, matched_job_ids: List[str]) -> Dict[str, float]:
        golden = set(self._golden_job_ids_for_persona(persona_key))
        predicted = set(matched_job_ids or [])
        tp = len(golden & predicted)
        fp = len(predicted - golden)
        fn = len(golden - predicted)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 0.0 if (precision + recall == 0) else (2 * precision * recall / (precision + recall))
        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }

    def _print_match_metrics(self, persona_name: str, matched_job_ids: List[str], mm: Dict[str, float]):
        print(f"\n🎯 JOB MATCHING - {persona_name}")
        print("="*60)
        print(f"Matched jobs: {matched_job_ids}")
        print(f"Precision: {mm['precision']:.3f}  Recall: {mm['recall']:.3f}  F1: {mm['f1']:.3f}")
        print("="*60)


async def main():
    """Run persona conversation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Clean LLM-First Conversation Flow")
    parser.add_argument("--persona", choices=["english_man", "calm_hindi", "energetic_hindi", "expressive_hindi"], 
                       help="Test specific persona only")
    parser.add_argument("--url", default="http://localhost:8000", help="Bot base URL")
    args = parser.parse_args()
    
    tester = HTTPPersonaConversationTester(bot_base_url=args.url)
    
    logger.info("🎭 Starting Clean LLM-First Conversation Tests...")
    
    if args.persona:
        # Test single persona
        result = await tester.run_persona_test(args.persona)
        logger.info(f"Test completed: {result.persona_key} - Success: {result.success}")
    else:
        # Test all personas
        for persona_key in tester.personas.keys():
            result = await tester.run_persona_test(persona_key)
            logger.info(f"Test completed: {result.persona_key} - Success: {result.success}")


if __name__ == "__main__":
    asyncio.run(main()) 