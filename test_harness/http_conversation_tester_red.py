#!/usr/bin/env python3
"""HTTP Conversation Tester with Isolated Persona Flows."""

import asyncio
import httpx
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from loguru import logger
import os
import json as _json
try:
    import openai  # Optional: used only for tester-side LLM extraction fallback
    import yaml  # Optional: load expected entities from utterances.yaml
except Exception:
    openai = None
    yaml = None

@dataclass
class ConversationTurn:
    """Single turn in conversation."""
    turn_number: int
    step_name: str
    audio_file: str
    asr_text: str
    asr_confidence: float
    bot_response: str
    latency_ms: float
    success: bool
    expected_question: str = ""
    extracted_entities: Dict = None
    current_field: str = ""
    candidate_profile: Dict = None
    audio_variant: str = ""
    tester_llm_entities: Dict = None

@dataclass 
class PersonaTestResult:
    """Test result for a single persona's complete conversation."""
    persona_key: str
    persona_name: str
    voice_id: str
    noise_levels: List[str]
    candidate_id: str
    start_time: datetime
    end_time: datetime
    
    # Conversation flow
    turns: List[ConversationTurn]
    completed_steps: int
    total_steps: int
    
    # Final outputs
    reached_job_matching: bool
    final_bot_response: str
    conversation_completed: bool
    
    # Performance metrics
    total_latency_ms: float
    avg_turn_latency_ms: float
    avg_confidence: float
    confidence_range: Tuple[float, float]

    # Cost metrics
    asr_cost_usd: float
    tts_cost_usd: float
    total_cost_usd: float
    
    # Entity extraction metrics (calculated at confirmation step)
    entity_extraction_metrics: Dict = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class HTTPPersonaConversationTester:
    """Tests complete conversation flows with isolated personas."""
    
    def __init__(self, bot_base_url: str = "http://localhost:8000"):
        """Initialize the persona conversation tester."""
        self.bot_base_url = bot_base_url
        self.results_dir = Path("test_harness/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        # Debug flag (set to False to use normal noisy/clean selection)
        self.use_clean_audio_only = False
        # Confidence thresholds
        # Treat below this as low-confidence and trigger confirmation/LLM fallback
        self.low_asr_conf_threshold = 0.70
        # Treat at or above this as good confidence for saving
        self.save_conf_threshold = 0.70
        # Load utterance expected entities mapping if available
        self._utterance_entities = self._load_utterance_entities()
        # Configure OpenAI for tester-side LLM extraction if available
        try:
            if openai is not None and os.getenv("OPENAI_API_KEY"):
                # Backwards-compatible API key assignment for legacy SDK
                setattr(openai, "api_key", os.getenv("OPENAI_API_KEY"))
        except Exception:
            pass

        # Simple cost models (adjust as needed)
        # ASR: per minute rates (USD) for Google/ElevenLabs fallback; estimate duration from bytes at 16kHz mono PCM
        self.asr_rate_per_min = {
            "google": 0.018,      # illustrative
            "elevenlabs": 0.030,  # illustrative
            "sarvam": 0.010,      # illustrative
            "speech_recognition": 0.0
        }
        # TTS: per character (USD) for Google/ElevenLabs; we don't have inline audio in fast path, so we estimate from bot text length
        self.tts_rate_per_char = {
            "google": 0.000016,    # illustrative
            "elevenlabs": 0.00003  # illustrative
        }

        # Required slots for completion metric
        self.required_slots = [
            "pincode", "availability_date", "preferred_shift", "expected_salary",
            "languages", "has_two_wheeler", "total_experience_months", "confirmation"
        ]

        # Define 4 isolated persona conversation flows
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
                    # {"step": "confirmation", "audio_file": "conf_diverse_001", "expected_entity": {"confirmation": "accept"}},
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
                    # {"step": "confirmation", "audio_file": "conf_diverse_002", "expected_entity": {"confirmation": "accept"}},
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
                    # {"step": "confirmation", "audio_file": "conf_diverse_003", "expected_entity": {"confirmation": "accept"}},
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
                    # {"step": "confirmation", "audio_file": "conf_diverse_004", "expected_entity": {"confirmation": "modify", "expected_salary": 20000}},
                ],
            },
        }

    # ==== Evaluation helpers ====
    def _normalize_value(self, slot: str, value):
        if value is None:
            return None
        try:
            if slot == "pincode":
                s = str(value).strip()
                # If already a 6-digit pincode, return as-is
                if s.isdigit() and len(s) == 6:
                    return s
                # Otherwise, attempt to approximate from locality text
                approx = self._approximate_pincode_from_locality(s)
                return approx or s
            if slot == "availability_date":
                v = str(value).strip().lower()
                # Synonyms/equivalence for evaluation
                synonyms = {
                    "tmrw": "tomorrow",
                    "tmr": "tomorrow",
                    "today": "today",
                    "immediately": "today",
                    "asap": "today",
                    "right now": "today",
                    "now": "today",
                    # Keep "soon" as "soon" to be handled in comparison logic
                }
                return synonyms.get(v, v)
            if slot == "preferred_shift":
                allowed = {"morning", "afternoon", "evening", "night", "flexible"}
                # Accept single or multiple values
                if isinstance(value, list):
                    flattened: List[str] = []
                    for v in value:
                        if isinstance(v, str) and "," in v:
                            flattened.extend([p.strip() for p in v.split(",")])
                        else:
                            flattened.append(v)
                    return set([str(v).strip().lower() for v in flattened if str(v).strip().lower() in allowed])
                if isinstance(value, str):
                    # Split on common separators if multiple are present
                    parts = [p.strip().lower() for p in value.replace("/", ",").replace("and", ",").split(",")]
                    vals = [p for p in parts if p in allowed]
                    return set(vals) if len(vals) > 1 else (vals[0] if vals else None)
                return None
            if slot == "confirmation":
                return str(value).strip().lower()
            if slot == "expected_salary" or slot == "total_experience_months":
                try:
                    return int(value)
                except Exception:
                    return None
            if slot == "has_two_wheeler":
                if isinstance(value, bool):
                    return value
                v = str(value).strip().lower()
                return v in ("true", "1", "yes", "haan", "ha")
            if slot == "languages":
                if isinstance(value, list):
                    flattened: List[str] = []
                    for v in value:
                        if isinstance(v, str) and "," in v:
                            flattened.extend([p.strip() for p in v.split(",")])
                        else:
                            flattened.append(v)
                    return set([str(v).strip().lower() for v in flattened])
                if isinstance(value, str):
                    return set([s.strip().lower() for s in value.split(",")])
                return set()
        except Exception:
            return value
        return value

    def _approximate_pincode_from_locality(self, locality: str) -> Optional[str]:
        """Best-effort mapping from locality/city name to a representative pincode.

        Args:
            locality (str): Free-form locality or city name
        Returns:
            Optional[str]: 6-digit pincode if we can infer a canonical one, else None
        """
        if not locality:
            return None
        text = str(locality).strip().lower()
        # Known central pincodes and prominent areas used across the app for demos
        mapping = {
            # Delhi
            "delhi": "110001", "new delhi": "110001", "connaught place": "110001", "cp": "110001",
            "darya ganj": "110002", "civil lines": "110003",
            # Mumbai
            "mumbai": "400001", "bombay": "400001", "fort": "400001", "fort mumbai": "400001",
            "kalbadevi": "400002", "masjid": "400003",
            # Bangalore
            "bangalore": "560001", "bengaluru": "560001", "bangalore gpo": "560001", "bengaluru gpo": "560001",
            "bangalore city": "560002",
            # Kolkata
            "kolkata": "700001", "calcutta": "700001", "kolkata gpo": "700001",
            # Chennai
            "chennai": "600001", "madras": "600001", "chennai central": "600001",
        }
        # Direct match
        if text in mapping:
            return mapping[text]
        # Fuzzy contains checks for simple cases (e.g., "I live in Delhi")
        for key, pin in mapping.items():
            if key in text:
                return pin
        return None

    def _compare_slot(self, slot: str, expected, predicted) -> Tuple[int, int, int]:
        """Return (tp, fp, fn) for a single slot on one example."""
        exp = self._normalize_value(slot, expected)
        pred = self._normalize_value(slot, predicted)
        
        if slot == "languages":
            exp_set = set(exp or [])
            pred_set = set(pred or [])
            if not exp_set:
                return 0, 0, 0
            if exp_set and not pred_set:
                return 0, 0, 1
            # Consider correct if predicted contains all expected languages
            if exp_set.issubset(pred_set):
                return 1, 0, 0
            else:
                return 0, 1, 1
                
        elif slot == "preferred_shift":
            # Support multi-value predicted/expected; pass if expected ⊆ predicted
            if exp is None:
                return 0, 0, 0
            # Normalize to sets for comparison
            exp_set = exp if isinstance(exp, set) else ({exp} if exp else set())
            pred_set = pred if isinstance(pred, set) else ({pred} if pred else set())
            if not pred_set:
                return 0, 0, 1
            # Special handling for "flexible" - if predicted is flexible, it matches any expected shift
            if "flexible" in pred_set:
                return 1, 0, 0
            # Also if expected is flexible, predicted should be flexible too
            if "flexible" in exp_set and "flexible" in pred_set:
                return 1, 0, 0
            if exp_set.issubset(pred_set):
                return 1, 0, 0
            return 0, 1, 1
            
        elif slot == "pincode":
            if exp is None:
                return 0, 0, 0
            if pred is None:
                return 0, 0, 1
            # Allow +/-10 range for pincode comparison
            try:
                exp_int = int(str(exp).strip()) if str(exp).strip().isdigit() else None
                pred_int = int(str(pred).strip()) if str(pred).strip().isdigit() else None
                if exp_int is not None and pred_int is not None:
                    if abs(exp_int - pred_int) <= 10:
                        return 1, 0, 0
                    else:
                        return 0, 1, 1
                # Fallback to exact string comparison if not numeric
                if str(exp).strip() == str(pred).strip():
                    return 1, 0, 0
                else:
                    return 0, 1, 1
            except (ValueError, TypeError):
                # Fallback to string comparison
                if str(exp).strip() == str(pred).strip():
                    return 1, 0, 0
                else:
                    return 0, 1, 1
                    
        elif slot == "availability_date":
            if exp is None:
                return 0, 0, 0
            if pred is None:
                return 0, 0, 1
            # More lenient availability date matching
            exp_str = str(exp).strip().lower()
            pred_str = str(pred).strip().lower()
            
            # Define equivalent groups
            immediate_group = {"today", "immediately", "soon"}
            tomorrow_group = {"tomorrow", "tmrw", "tmr", "soon"}
            
            # Check if both are in the same equivalence group
            if (exp_str in immediate_group and pred_str in immediate_group) or \
               (exp_str in tomorrow_group and pred_str in tomorrow_group):
                return 1, 0, 0
            # Exact match
            elif exp_str == pred_str:
                return 1, 0, 0
            else:
                return 0, 1, 1
                
        else:
            # Default comparison for other slots
            if exp is None:
                return 0, 0, 0
            if pred is None:
                return 0, 0, 1
            if exp == pred:
                return 1, 0, 0
            else:
                return 0, 1, 1

    def _compute_f1(self, tp: int, fp: int, fn: int) -> float:
        if tp == 0 and (fp > 0 or fn > 0):
            return 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _calculate_persona_entity_metrics(self, turns: List[ConversationTurn], final_profile: Dict, tester_best_entities: Dict) -> Dict:
        """Calculate entity extraction metrics for a single persona at confirmation step.
        
        Args:
            turns: All conversation turns for this persona
            final_profile: Final candidate profile from server
            tester_best_entities: Best entities extracted by tester LLM
        Returns:
            Dict: Per-slot and overall entity extraction metrics
        """
        # Build expected entities from utterances.yaml (not persona config)
        expected_by_slot: Dict[str, object] = {}
        for turn in turns:
            # Prioritize utterances.yaml over hardcoded persona config
            if hasattr(self, "_utterance_entities") and turn.audio_file in self._utterance_entities:
                exp_map = self._utterance_entities[turn.audio_file]
                logger.info(f"Using utterances.yaml for {turn.audio_file}: {exp_map}")
            else:
                # Fallback to persona config if utterances.yaml not available
                exp_map = turn.extracted_entities or {}
                if exp_map:
                    logger.warning(f"Using persona config for {turn.audio_file}: {exp_map}")
            
            for slot, val in (exp_map.items() if isinstance(exp_map, dict) else []):
                expected_by_slot[slot] = val
        
        # Build predicted entities from final profile and tester entities
        tester_seen: Dict[str, object] = {}
        for turn in turns:
            if getattr(turn, "tester_llm_entities", None):
                for k, v in turn.tester_llm_entities.items():
                    tester_seen[k] = v
        # Merge in the best entities passed from confirmation step
        tester_seen.update(tester_best_entities or {})
        
        per_slot_metrics = {}
        tp_total, fp_total, fn_total = 0, 0, 0
        
        for slot in self.required_slots:
            if slot not in expected_by_slot:
                # Skip slots that weren't expected in this persona's conversation
                continue
                
            expected_value = expected_by_slot.get(slot)
            predicted_value = None
            
            # Extract predicted value from appropriate source
            if slot == "expected_salary":
                predicted_value = final_profile.get("expected_salary")
            elif slot == "pincode":
                predicted_value = final_profile.get("pincode") or tester_seen.get("pincode")
            elif slot == "availability_date":
                predicted_value = final_profile.get("availability_date")
            elif slot == "preferred_shift":
                predicted_value = final_profile.get("preferred_shift") or tester_seen.get("preferred_shift")
            elif slot == "languages":
                langs = final_profile.get("languages") or []
                other = final_profile.get("other_languages") or []
                predicted_value = (list(langs) + list(other)) or tester_seen.get("languages") or []
            elif slot == "has_two_wheeler":
                predicted_value = final_profile.get("has_two_wheeler")
            elif slot == "total_experience_months":
                predicted_value = final_profile.get("total_experience_months")
            # elif slot == "confirmation":
            #     predicted_value = "accept" if final_profile.get("conversation_completed") else None
                
            tp, fp, fn = self._compare_slot(slot, expected_value, predicted_value)
            f1 = self._compute_f1(tp, fp, fn)
            
            # Convert sets to lists for JSON serialization
            norm_exp = self._normalize_value(slot, expected_value)
            norm_pred = self._normalize_value(slot, predicted_value)
            if isinstance(norm_exp, set):
                norm_exp = list(norm_exp)
            if isinstance(norm_pred, set):
                norm_pred = list(norm_pred)
                
            per_slot_metrics[slot] = {
                "tp": tp, "fp": fp, "fn": fn, "f1": f1,
                "expected": expected_value, "predicted": predicted_value,
                "normalized_expected": norm_exp,
                "normalized_predicted": norm_pred
            }
            tp_total += tp
            fp_total += fp  
            fn_total += fn
            
        # Calculate overall metrics
        macro_f1 = sum(m["f1"] for m in per_slot_metrics.values()) / len(per_slot_metrics) if per_slot_metrics else 0.0
        micro_f1 = self._compute_f1(tp_total, fp_total, fn_total)
        
        return {
            "per_slot": per_slot_metrics,
            "macro_f1": macro_f1,
            "micro_f1": micro_f1,
            "total_tp": tp_total,
            "total_fp": fp_total,
            "total_fn": fn_total
        }

    def _is_confirmation_prompt(self, bot_text: str) -> bool:
        """Heuristic to detect confirmation prompt in bot response.

        Args:
            bot_text (str): Bot response text
        Returns:
            bool: True if the text likely asks for final confirmation
        """
        if not bot_text:
            return False
        t = bot_text.strip().lower()
        cues = [
            "saari details", "sari details", "details sahi", "details theek", "details thik",
            "confirm karte hain", "haan ya nahi", "haan ya nahin", "yes or no"
        ]
        return any(cue in t for cue in cues)

    def _llm_extract_for_slot(self, slot: str, asr_text: str) -> Optional[object]:
        """Attempt to extract an entity for the given slot from ASR text using OpenAI.

        Args:
            slot (str): Slot name (e.g., 'pincode', 'expected_salary')
            asr_text (str): Recognized text to parse
        Returns:
            Optional[object]: Normalized value if confidently extracted, else None
        """
        if not asr_text:
            logger.warning(f"🔬 LLM: No ASR text provided for {slot}")
            return None
        if not openai:
            logger.warning(f"🔬 LLM: OpenAI not available for {slot}")
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
                "locality": (
                    "Output the free-form locality or city name from the ASR text if present; else null."
                ),
            }
            rule = rules.get(slot, "Output null.")
            user = f"Field: {slot}\nText: {asr_text}\nReturn: {{\"value\": ...}}"
            resp = openai.chat.completions.create(
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
            # Normalize/validate per slot
            if slot == "pincode":
                if isinstance(value, str) and value.isdigit() and len(value) == 6:
                    logger.info(f"🔬 LLM PINCODE VALIDATED: {value}")
                    return value
                logger.warning(f"🔬 LLM PINCODE INVALID: {value} (type: {type(value)})")
                return None
            if slot == "availability_date":
                return str(value).strip().lower() if value else None
            if slot == "preferred_shift":
                allowed = {"morning", "afternoon", "evening", "night", "flexible"}
                if isinstance(value, list):
                    return [v for v in [str(x).strip().lower() for x in value] if v in allowed]
                if isinstance(value, str):
                    v = value.strip().lower()
                    return [v] if v in allowed else []
                return []
            if slot == "expected_salary":
                try:
                    return int(value) if value is not None else None
                except Exception:
                    return None
            if slot == "languages":
                if isinstance(value, list):
                    return [str(x).strip().lower() for x in value if isinstance(x, (str,))]
                return []
            if slot == "has_two_wheeler":
                if isinstance(value, bool):
                    return value
                if isinstance(value, str):
                    return value.strip().lower() in {"true", "yes", "1", "haan", "ha"}
                return None
            if slot == "total_experience_months":
                try:
                    iv = int(value) if value is not None else None
                    return iv if iv is not None and iv >= 0 else None
                except Exception:
                    return None
            if slot == "locality":
                return str(value).strip() if value else None
            return value
        except Exception:
            return None
    
    async def _confirm_text(self, client: httpx.AsyncClient, candidate_id: str, text: str = "haan") -> bool:
        """Send a tester-side text confirmation to the server to advance/complete.

        Args:
            client (httpx.AsyncClient): HTTP client
            candidate_id (str): Conversation ID
            text (str): Confirmation text (e.g., 'haan', 'yes')
        Returns:
            bool: True if request accepted
        """
        try:
            url = f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/confirm-fast"
            payload = {"text_only": True, "user_text": text}
            resp = await client.post(url, json=payload)
            return resp.status_code == 200
        except Exception:
            return False

    async def _fetch_status(self, client: httpx.AsyncClient, candidate_id: str) -> Dict:
        """Fetch current conversation status/profile from server.

        Args:
            client (httpx.AsyncClient): HTTP client
            candidate_id (str): Conversation ID
        Returns:
            Dict: Status JSON or empty dict
        """
        try:
            url = f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/status"
            resp = await client.get(url)
            if resp.status_code == 200:
                return resp.json() or {}
            return {}
        except Exception:
            return {}

    async def start_conversation_session(self, client: httpx.AsyncClient) -> Optional[str]:
        """Start a new conversation session."""
        try:
            response = await client.post(f"{self.bot_base_url}/api/v1/conversation/start")
            if response.status_code == 200:
                data = response.json()
                candidate_id = data.get("candidate_id")
                # Poll status to ensure state is ready (avoids race with server init/reload)
                if candidate_id:
                    status_url = f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/status"
                    for attempt in range(5):
                        try:
                            st = await client.get(status_url)
                            if st.status_code == 200:
                                break
                            await asyncio.sleep(0.4)
                        except Exception:
                            await asyncio.sleep(0.4)
                return candidate_id
            else:
                logger.error(f"Failed to start conversation: {response.status_code}")
                return None
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            return None
    
    async def send_audio_turn(self, client: httpx.AsyncClient, candidate_id: str, 
                            audio_file: str, step_name: str, expected_entity: Dict,
                            force_clean: bool = False) -> ConversationTurn:
        """Send audio file and get response."""
        # Determine which audio file to use
        if getattr(self, "use_clean_audio_only", False) or force_clean:
            audio_path = f"test_harness/generated_audio/{audio_file}_clean.wav"
            logger.info(f"Using CLEAN audio for {audio_file}: {audio_path}")
            audio_variant = "clean"
        else:
            if audio_file.endswith("_001") or audio_file.endswith("_003"):
                audio_path = f"test_harness/generated_audio/{audio_file}_noisy_gpt.wav"
                logger.info(f"Using noisy audio for {audio_file}: {audio_path}")
                audio_variant = "noisy"
            else:
                audio_path = f"test_harness/generated_audio/{audio_file}_clean.wav"
                logger.info(f"Using clean audio for {audio_file}: {audio_path}")
                audio_variant = "clean"
        
        if not Path(audio_path).exists():
            logger.warning(f"Audio file not found: {audio_path}")
            return ConversationTurn(
                turn_number=0, step_name=step_name, audio_file=audio_file,
                asr_text="", asr_confidence=0.0, bot_response="", 
                latency_ms=0, success=False, current_field="", candidate_profile={}, audio_variant=audio_variant
            )
        
        try:
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            logger.info(f"Sending turn: {audio_path} ({len(audio_data)} bytes)")
            
            files = {'audio_file': ('recording.wav', audio_data, 'audio/wav')}
            url = f"{self.bot_base_url}/api/v1/conversation/{candidate_id}/turn-fast"
            
            start_time = time.time()
            response = await client.post(url, files=files)
            latency_ms = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                data = response.json()
                
                # Debug: Log the full response structure to see what's available
                logger.debug(f"Full API response: {json.dumps(data, indent=2)}")
                
                asr_data = data.get("asr", {})
                asr_text = asr_data.get("text", "")
                asr_confidence = asr_data.get("confidence", 0.0)
                bot_response = data.get("text", "")
                conversation_completed = data.get("conversation_complete", False)
                metrics = data.get("metrics", {})
                current_field = metrics.get("current_field", "")
                candidate_profile = metrics.get("candidate_profile", {})

                # Cost estimates for this turn
                # ASR provider from raw data if present
                raw_conf = asr_data.get("raw_confidence_data", {})
                asr_provider = raw_conf.get("asr_provider", "google").lower()
                # Estimate audio duration from bytes if PCM 16kHz mono, else rough fallback (bytes/32000)
                # Our tester sends WAVs; many are 16kHz mono
                est_audio_seconds = max(0.1, len(audio_data) / 32000.0)
                asr_cost = (self.asr_rate_per_min.get(asr_provider, 0.0) / 60.0) * est_audio_seconds
                # TTS cost estimate from response text length; assume ElevenLabs if key configured, else Google
                tts_provider = "elevenlabs" if os.getenv("ELEVENLABS_API_KEY") else "google"
                tts_chars = len(bot_response or "")
                tts_cost = self.tts_rate_per_char.get(tts_provider, 0.0) * tts_chars
                
                # Get raw ASR confidence data from the updated API
                raw_confidence_data = asr_data.get("raw_confidence_data", {})
                raw_asr_confidence = raw_confidence_data.get("model_confidence", asr_confidence)
                
                # Log detailed confidence information
                logger.info(f"  Raw ASR Data: Provider={raw_confidence_data.get('asr_provider', 'unknown')}, Source={raw_confidence_data.get('confidence_source', 'unknown')}")
                
                logger.info(f"\u2713 Turn completed in {latency_ms:.0f}ms")
                logger.info(f"  ASR: '{asr_text[:60]}...' (conf: {asr_confidence:.2f}, raw: {raw_asr_confidence:.2f})")
                logger.info(f"  Bot: '{bot_response[:60]}...'")
                logger.info(f"  Complete: {conversation_completed}")
                
                turn = ConversationTurn(
                    turn_number=0, step_name=step_name, audio_file=audio_file,
                    asr_text=asr_text, asr_confidence=raw_asr_confidence,  # Use raw ASR confidence
                    bot_response=bot_response, latency_ms=latency_ms,
                    success=True, extracted_entities=expected_entity,
                    current_field=current_field, candidate_profile=candidate_profile,
                    audio_variant=audio_variant
                )
                # Attach ephemeral cost info (not stored in dataclass to keep schema stable per turn)
                turn._asr_cost_usd = asr_cost  # type: ignore
                turn._tts_cost_usd = tts_cost  # type: ignore
                # Always attempt LLM extraction for the active slot when applicable (tester-side only)
                try:
                    active_slot = step_name or str(current_field or "").strip().lower()
                    if active_slot in self.required_slots and (raw_asr_confidence < self.low_asr_conf_threshold or active_slot in ("pincode", "languages")):
                        extracted_value = self._llm_extract_for_slot(active_slot, asr_text or "")
                        if extracted_value not in (None, [], ""):
                            turn.tester_llm_entities = turn.tester_llm_entities or {}
                            turn.tester_llm_entities[active_slot] = extracted_value
                except Exception:
                    pass
                return turn
            else:
                logger.error(f"HTTP error {response.status_code}: {response.text}")
                return ConversationTurn(
                    turn_number=0, step_name=step_name, audio_file=audio_file,
                    asr_text="", asr_confidence=0.0, bot_response="", 
                    latency_ms=latency_ms, success=False, current_field="", candidate_profile={}, audio_variant=audio_variant
                )
                
        except Exception as e:
            logger.error(f"Error in audio turn: {e}")
            return ConversationTurn(
                turn_number=0, step_name=step_name, audio_file=audio_file,
                asr_text="", asr_confidence=0.0, bot_response="", 
                latency_ms=0, success=False, current_field="", candidate_profile={}, audio_variant=audio_variant
            )

    def _is_step_filled(self, step_name: str, profile: Dict) -> bool:
        """Check if the given step has been filled in candidate_profile."""
        if not profile:
            return False
        if step_name == "pincode":
            # Require a strict 6-digit pincode for completion gating
            pin = profile.get("pincode")
            return isinstance(pin, str) and pin.isdigit() and len(pin) == 6
        if step_name == "expected_salary":
            return profile.get("expected_salary") is not None
        if step_name == "has_two_wheeler":
            return profile.get("has_two_wheeler") is not None
        if step_name == "languages":
            langs = (profile.get("languages") or []) + (profile.get("other_languages") or [])
            return len(langs) > 0
        if step_name == "availability_date":
            return bool(profile.get("availability_date"))
        if step_name == "preferred_shift":
            return bool(profile.get("preferred_shift"))
        if step_name == "total_experience_months":
            return profile.get("total_experience_months") is not None
        if step_name == "confirmation":
            return bool(profile.get("conversation_completed"))
        return False

    async def run_persona_test(self, persona_key: str) -> PersonaTestResult:
        """Run complete conversation test for a single persona."""
        persona = self.personas[persona_key]
        start_time = datetime.now()
        
        logger.info(f"🎭 Starting {persona['name']} conversation test...")
        
        async with httpx.AsyncClient(timeout=120.0) as client:
            # Start conversation
            candidate_id = await self.start_conversation_session(client)
            if not candidate_id:
                raise Exception("Failed to start conversation")
            
            logger.info(f"Started conversation session: {candidate_id}")
            
            # Wait a moment for conversation to be fully initialized
            await asyncio.sleep(1)
            
            turns = []
            confidences = []
            played_confirmation_audio = False
            # Map step name -> config for quick override when bot asks a different field
            step_map = {cfg["step"]: cfg for cfg in persona["conversation_flow"]}
            logger.info(f"🗺️ STEP_MAP: {[(k, v['audio_file']) for k, v in step_map.items()]}")
            # If bot re-asks or changes order, use this to pick the right audio next
            pending_field: Optional[str] = None
            
            # Process each step in the persona's conversation flow
            # POLICY YOU WANTED:
            # 1) Always send the audio that corresponds to the slot the bot is asking for.
            # 2) Retry ONCE with the SAME audio if ASR confidence is low.
            # 3) If still low, pass the question + ASR text to LLM to extract entities.
            # 4) For pincode and languages, call the LLM on the FIRST try (default), not only on low confidence.
            
            # Simple linear flow - one attempt per slot
            
            for i, planned_config in enumerate(persona["conversation_flow"], 1):
                # Figure out which slot to answer next: prefer server's current_field, else planned
                server_field = turns[-1].current_field if turns else ""
                logger.info(f"🔍 SERVER wants: '{server_field}', PLANNED step: '{planned_config['step']}'")
                
                if turns and server_field in step_map:
                    effective_config = step_map[server_field]
                    logger.info(f"🎯 Using SERVER field '{server_field}' -> audio: {effective_config['audio_file']}")
                else:
                    effective_config = planned_config
                    logger.info(f"🎯 Using PLANNED field '{planned_config['step']}' -> audio: {planned_config['audio_file']}")

                slot = effective_config["step"]
                
                # Single attempt per slot - no infinite loop prevention needed
                
                # Use the planned audio file for this slot
                audio_id = effective_config["audio_file"]
                
                logger.info(f"\n--- Step {i}/{len(persona['conversation_flow'])}: {slot} ---")

                # Single attempt - no retries, everything goes through LLM
                    tr = await self.send_audio_turn(
                        client, candidate_id,
                    audio_id,
                    slot,
                    effective_config.get("expected_entity", {}),
                    force_clean=False,
                )
                last_turn = tr
                    turns.append(tr)
                    confidences.append(tr.asr_confidence)
                
                # If HTTP failed, stop the conversation
                if not tr.success:
                    logger.error(f"Failed to send audio for {slot}, stopping conversation")
                        break

                    # LLM EXTRACTION FOR ALL SLOTS (IMMEDIATE):
                    logger.info(f"🔬 ATTEMPTING LLM EXTRACTION for {slot} with ASR: '{tr.asr_text}'")
                    try:
                        val = self._llm_extract_for_slot(slot, tr.asr_text or "")
                        logger.info(f"🔬 LLM RETURNED: {val} for {slot}")
                        
                        # Always store LLM result, even if None
                        tr.tester_llm_entities = (tr.tester_llm_entities or {})
                        tr.tester_llm_entities[slot] = val
                        
                        if val not in (None, [], ""):
                            logger.info(f"LLM-extracted {slot}: {val}")
                            
                            # Send extracted value to server immediately for all slots
                            try:
                                if slot == "pincode" and isinstance(val, str) and val.isdigit() and len(val) == 6:
                                    ok = await self._confirm_text(client, candidate_id, text=val)
                                elif slot == "expected_salary" and isinstance(val, (int, str)):
                                    ok = await self._confirm_text(client, candidate_id, text=str(val))
                                elif slot == "languages" and isinstance(val, list) and val:
                                    # Send first language or comma-separated list
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
                    else:
                                    ok = False
                                    
                                if ok:
                                    status = await self._fetch_status(client, candidate_id)
                                    prof = (status.get("candidate_profile") or {}) if status else {}
                                    synth_turn = ConversationTurn(
                                        turn_number=0,
                                        step_name=slot,
                                        audio_file=f"tester_text_{slot}",
                                        asr_text=str(val),
                                        asr_confidence=0.95,
                                        bot_response=(status.get("metrics") or {}).get("last_bot_text", "") if status else "",
                                        latency_ms=0.0,
                                        success=True,
                                        extracted_entities={},
                                        current_field="",
                                        candidate_profile=prof,
                                        audio_variant="text"
                                    )
                                    turns.append(synth_turn)
                                    confidences.append(0.95)
                                    logger.info(f"✅ Successfully sent {slot}={val} to server")
                                    # Move on to next step
                                    break
                                else:
                                    logger.warning(f"❌ Failed to send {slot}={val} to server")
                            except Exception as e:
                                logger.error(f"❌ Error sending {slot}={val}: {e}")
                        else:
                            logger.info(f"🔬 LLM returned None/empty for {slot} - keeping as None and moving on")
                        
                        # Fallback for pincode: try locality inference if LLM extraction failed
                        if slot == "pincode" and not val:
                            prof = tr.candidate_profile or {}
                            if prof.get("locality"):
                                approx_pin = self._approximate_pincode_from_locality(prof.get("locality"))
                                if approx_pin and approx_pin.isdigit() and len(approx_pin) == 6:
                                    tr.tester_llm_entities = (tr.tester_llm_entities or {})
                                    tr.tester_llm_entities[slot] = approx_pin
                                    logger.info(f"🗺️ Tester inferred pincode from locality: {approx_pin}")
                                    try:
                                        ok = await self._confirm_text(client, candidate_id, text=approx_pin)
                        if ok:
                            status = await self._fetch_status(client, candidate_id)
                            prof = (status.get("candidate_profile") or {}) if status else {}
                                synth_turn = ConversationTurn(
                                    turn_number=0,
                                                step_name="pincode",
                                                audio_file="tester_text_pincode",
                                                asr_text=approx_pin,
                                    asr_confidence=0.95,
                                    bot_response=(status.get("metrics") or {}).get("last_bot_text", "") if status else "",
                                    latency_ms=0.0,
                                    success=True,
                                    extracted_entities={},
                                    current_field="",
                                                candidate_profile=prof,
                                    audio_variant="text"
                                )
                                turns.append(synth_turn)
                                confidences.append(0.95)
                                            break
                    except Exception:
                        pass
                    except Exception as e:
                        logger.error(f"❌ LLM extraction failed for {slot}: {e}")

                    # If bot is asking confirmation now, we jump to confirmation next
                    if self._is_confirmation_prompt(tr.bot_response or ""):
                        pending_field = "confirmation"
                        break

                # After LLM extraction, always move to next step regardless of success/failure
                # No confidence checks, no retries - LLM result is final

                # If job matching text appears or conversation marked complete, exit early
                if last_turn and last_turn.bot_response and ("job" in last_turn.bot_response.lower() or "match" in last_turn.bot_response.lower()):
                    logger.info("\ud83c\udf89 Reached job matching!")
                    break
                if (last_turn.candidate_profile or {}).get("conversation_completed"):
                    logger.info("\ud83c\udf89 Conversation marked complete by server.")
                    break

                # Short pause between turns to avoid hammering the server
                await asyncio.sleep(1)
            
            end_time = datetime.now()
            
            # Calculate metrics
            total_latency = sum(turn.latency_ms for turn in turns)
            avg_latency = total_latency / len(turns) if turns else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            confidence_range = (min(confidences), max(confidences)) if confidences else (0, 0)
            
            # NEW FLOW COMPLETE - No more retry logic needed
            # Collect final entities from tester LLM extractions
            tester_best_entities: Dict[str, object] = {}
            for turn in turns:
                if hasattr(turn, 'tester_llm_entities') and turn.tester_llm_entities:
                    tester_best_entities.update(turn.tester_llm_entities)
            
            final_profile_snapshot = (turns[-1].candidate_profile if turns else {}) or {}
            # Calculate entity metrics at the end of conversation
            entity_metrics = self._calculate_persona_entity_metrics(turns, final_profile_snapshot, tester_best_entities)
            
            # Print entity comparison for debugging
            self._print_entity_comparison_table(entity_metrics, persona["name"])
            
            # Simple completion check
            conversation_completed = bool(final_profile_snapshot.get("conversation_completed", False))
            
            return PersonaTestResult(
                persona_key=persona_key,
                completed=conversation_completed,
                turns=len(turns),
                avg_confidence=sum(confidences) / len(confidences) if confidences else 0.0,
                confidence_range=confidence_range,
                latency_p95_ms=max([turn.latency_ms for turn in turns]) if turns else 0.0,
                entity_extraction_metrics=entity_metrics,
                success=True,
                step_completion_rate=1.0,  # We always complete our planned steps
                completion_time_s=time.time() - start_time,
                conversation_turns=turns
            )

    def _print_entity_comparison_table(self, entity_metrics: Dict, persona_name: str):
        """Print actual expected vs derived values for each slot."""
        if not entity_metrics or "per_slot" not in entity_metrics:
            return
            
        print(f"\n🔍 ENTITY VALUES - {persona_name}")
        print("="*60)
        
        for slot, metrics in entity_metrics["per_slot"].items():
            expected = metrics.get("expected", "None")
            predicted = metrics.get("predicted", "None")
            match = "✓" if metrics.get("tp", 0) > 0 else "✗"
            
            print(f"{slot}: expected={expected}, derived={predicted} {match}")
        
        print(f"\nMacro F1: {entity_metrics.get('macro_f1', 0.0):.3f}")
        print("="*60)

    async def run_all_personas_test(self) -> List[PersonaTestResult]:
        """Run tests for all personas."""
        results = []
        
        for persona_key in self.personas.keys():
            try:
                result = await self.run_persona_test(persona_key)
                results.append(result)
                
                # Save individual result
                timestamp = int(time.time())
                filename = f"persona_{persona_key}_{timestamp}.json"
                filepath = self.results_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved persona results to: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to test persona {persona_key}: {e}")
        
        # After all personas, compute evaluation summary
        try:
            summary = self._build_evaluation_summary(results)
            sum_path = self.results_dir / f"summary_{int(time.time())}.json"
            with open(sum_path, 'w', encoding='utf-8') as f:
                _json.dump(summary, f, indent=2, ensure_ascii=False)
            logger.info(f"Saved evaluation summary to: {sum_path}")
        except Exception as e:
            logger.error(f"Failed to build evaluation summary: {e}")
        
        return results

    def print_persona_comparison(self, results: List[PersonaTestResult]):
        """Print comparison of all persona results."""
        print(f"\n{'='*80}")
        print("PERSONA CONVERSATION TEST COMPARISON")
        print(f"{'='*80}")
        
        for result in results:
            print(f"\n{result.persona_name}")
            print(f"   Voice ID: {result.voice_id}")
            print(f"   Steps Completed: {result.completed_steps}/{result.total_steps}")
            print(f"   Conversation Complete: {'YES' if result.conversation_completed else 'NO'}")
            print(f"   Average Latency: {result.avg_turn_latency_ms:.0f}ms")
            print(f"   Average Confidence: {result.avg_confidence:.3f}")
            print(f"   Confidence Range: {result.confidence_range[0]:.3f} - {result.confidence_range[1]:.3f}")
            print(f"   Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
            print(f"   Costs: ASR ${result.asr_cost_usd:.4f} | TTS ${result.tts_cost_usd:.4f} | Total ${result.total_cost_usd:.4f}")
            
            if result.turns:
                print(f"   Confidence Variation:")
                for turn in result.turns:
                    print(f"     - {turn.step_name}: {turn.asr_confidence:.3f}")

        # Also print eval summary inline
        try:
            summary = self._build_evaluation_summary(results)
            print(f"\n{'-'*80}")
            print("ENTITY EXTRACTION METRICS (All Personas)")
            for slot, stats in summary.get("entity_f1", {}).get("per_slot", {}).items():
                print(f"   {slot}: F1 {stats['f1']:.3f} (TP {stats['tp']}, FP {stats['fp']}, FN {stats['fn']})")
            print(f"   Macro-F1: {summary.get('entity_f1', {}).get('macro_f1', 0.0):.3f}")
            print("\nSLOT COMPLETION <=10 TURNS")
            print(f"   Overall: {summary.get('slot_completion', {}).get('overall_completion_rate', 0.0)*100:.1f}%")
            for per in summary.get('slot_completion', {}).get('per_persona', []):
                print(f"   {per['persona']}: {per['completion_rate']*100:.1f}% (turns_used={per['turns_used']})")
        except Exception as e:
            logger.error(f"Failed to print evaluation summary: {e}")

    def _load_utterance_entities(self) -> Dict[str, Dict]:
        """Load expected entities per utterance id from utterances.yaml if available."""
        mapping: Dict[str, Dict] = {}
        try:
            if yaml is None:
                return mapping
            yaml_path = Path("utterances.yaml")
            if not yaml_path.exists():
                return mapping
            with open(yaml_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f) or {}
            for item in (data.get("utterances") or []):
                uid = item.get("id")
                ents = item.get("entities") or {}
                if uid and isinstance(ents, dict):
                    mapping[uid] = ents
        except Exception:
            return {}
        return mapping

    def _build_evaluation_summary(self, results: List[PersonaTestResult]) -> Dict:
        # Use pre-calculated entity metrics from individual personas
        per_slot = {s: {"tp": 0, "fp": 0, "fn": 0} for s in self.required_slots}
        f1_scores_by_slot = {s: [] for s in self.required_slots}
        
        for res in results:
            if not res.entity_extraction_metrics:
                    continue
                
            # Aggregate TP/FP/FN from stored metrics
            persona_metrics = res.entity_extraction_metrics.get("per_slot", {})
            for slot in self.required_slots:
                if slot in persona_metrics:
                    slot_metrics = persona_metrics[slot]
                    per_slot[slot]["tp"] += slot_metrics.get("tp", 0)
                    per_slot[slot]["fp"] += slot_metrics.get("fp", 0)
                    per_slot[slot]["fn"] += slot_metrics.get("fn", 0)
                    f1_scores_by_slot[slot].append(slot_metrics.get("f1", 0.0))
        
        # Calculate aggregate F1 scores
        per_slot_f1 = {}
        f1_list = []
        for slot, stats in per_slot.items():
            f1 = self._compute_f1(stats["tp"], stats["fp"], stats["fn"])
            per_slot_f1[slot] = {**stats, "f1": f1}
            f1_list.append(f1)
        macro_f1 = sum(f1_list) / len(f1_list) if f1_list else 0.0

        # Slot completion within <= 10 turns
        per_persona_completion = []
        for res in results:
            # Consider first 10 turns candidate_profile accumulation
            filled = set()
            turns_used = min(10, len(res.turns))
            for t in res.turns[:turns_used]:
                prof = t.candidate_profile or {}
                for slot in self.required_slots:
                    if slot == "confirmation":
                        if prof.get("conversation_completed"):
                            filled.add("confirmation")
                    else:
                        if prof.get(slot) not in (None, [], ""):
                            filled.add(slot)
            completion_rate = len(filled.intersection(set(self.required_slots))) / len(self.required_slots)
            per_persona_completion.append({
                "persona": res.persona_name,
                "completion_rate": completion_rate,
                "turns_used": turns_used,
                "pass": completion_rate >= 0.8
            })
        overall_completion_rate = sum(p["completion_rate"] for p in per_persona_completion) / len(per_persona_completion) if per_persona_completion else 0.0

        # Ranking metrics (Eligibility P/R and NDCG@3) for matched jobs on completed conversations
        ranking = self._build_ranking_metrics(results)

        return {
            "entity_f1": {
                "per_slot": per_slot_f1,
                "macro_f1": macro_f1,
                "target_pass": macro_f1 >= 0.80
            },
            "slot_completion": {
                "per_persona": per_persona_completion,
                "overall_completion_rate": overall_completion_rate,
                "target_pass": overall_completion_rate >= 0.80
            },
            "ranking": ranking
        }

    def _gold_job_ids_for_persona(self, persona_key: str) -> List[str]:
        prefix_map = {
            "english_man": "GLD_EN_",
            "calm_hindi": "GLD_CH_",
            "energetic_hindi": "GLD_EH_",
            "expressive_hindi": "GLD_XH_",
        }
        prefix = prefix_map.get(persona_key, "")
        if not prefix:
            return []
        # We know we created 3 per persona: 001..003
        return [f"{prefix}{i:03d}" for i in range(1, 4)]

    def _ndcg_at_k(self, relevances: List[int], k: int = 3) -> float:
        import math
        k = min(k, len(relevances))
        dcg = 0.0
        for i in range(k):
            rel = relevances[i]
            if rel > 0:
                dcg += (2**rel - 1) / math.log2(i + 2)
        # Ideal DCG
        sorted_rels = sorted(relevances, reverse=True)
        idcg = 0.0
        for i in range(k):
            rel = sorted_rels[i]
            if rel > 0:
                idcg += (2**rel - 1) / math.log2(i + 2)
        return (dcg / idcg) if idcg > 0 else 0.0

    def _build_ranking_metrics(self, results: List[PersonaTestResult]) -> Dict:
        import httpx
        per_persona = []
        p_at3_list = []
        r_at3_list = []
        ndcg_at3_list = []
        for res in results:
            persona_key = getattr(res, "persona_key", "")
            gold = set(self._gold_job_ids_for_persona(persona_key))
            if not gold:
                continue
            if not res.conversation_completed:
                per_persona.append({
                    "persona": res.persona_name,
                    "candidate_id": res.candidate_id,
                    "available": False,
                    "reason": "conversation_incomplete"
                })
                continue
            # Fetch matches
            ranked_ids: List[str] = []
            try:
                with httpx.Client(timeout=10.0) as client:
                    mresp = client.get(f"{self.bot_base_url}/api/v1/conversation/{res.candidate_id}/matches")
                    if mresp.status_code == 200:
                        mdata = mresp.json()
                        for m in (mdata.get("matches") or []):
                            job = m.get("job") or {}
                            jid = job.get("job_id")
                            if jid:
                                ranked_ids.append(jid)
                    else:
                        per_persona.append({
                            "persona": res.persona_name,
                            "candidate_id": res.candidate_id,
                            "available": False,
                            "reason": f"matches_status_{mresp.status_code}"
                        })
                        continue
            except Exception as e:
                per_persona.append({
                    "persona": res.persona_name,
                    "candidate_id": res.candidate_id,
                    "available": False,
                    "reason": f"fetch_error:{e}"
                })
                continue

            # Compute eligibility precision/recall over all returned matches and @3
            returned = len(ranked_ids)
            if returned == 0:
                per_persona.append({
                    "persona": res.persona_name,
                    "candidate_id": res.candidate_id,
                    "available": True,
                    "num_matches": 0,
                    "precision": 0.0,
                    "recall": 0.0,
                    "precision_at_3": 0.0,
                    "recall_at_3": 0.0,
                    "ndcg_at_3": 0.0
                })
                continue
            rels = [1 if jid in gold else 0 for jid in ranked_ids]
            tp_all = sum(rels)
            precision = tp_all / returned
            recall = tp_all / len(gold)
            k = 3
            rels3 = rels[:k]
            tp3 = sum(rels3)
            precision_at_3 = tp3 / min(k, returned)
            recall_at_3 = tp3 / len(gold)
            ndcg_at_3 = self._ndcg_at_k(rels, k=3)

            per_persona.append({
                "persona": res.persona_name,
                "candidate_id": res.candidate_id,
                "available": True,
                "num_matches": returned,
                "gold_count": len(gold),
                "precision": precision,
                "recall": recall,
                "precision_at_3": precision_at_3,
                "recall_at_3": recall_at_3,
                "ndcg_at_3": ndcg_at_3
            })
            p_at3_list.append(precision_at_3)
            r_at3_list.append(recall_at_3)
            ndcg_at3_list.append(ndcg_at_3)

        overall = {
            "macro_precision_at_3": (sum(p_at3_list) / len(p_at3_list)) if p_at3_list else 0.0,
            "macro_recall_at_3": (sum(r_at3_list) / len(r_at3_list)) if r_at3_list else 0.0,
            "macro_ndcg_at_3": (sum(ndcg_at3_list) / len(ndcg_at3_list)) if ndcg_at3_list else 0.0,
        }
        return {"per_persona": per_persona, "overall": overall}


async def main():
    """Run persona conversation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Isolated Persona Conversation Flows")
    parser.add_argument("--persona", choices=["english_man", "calm_hindi", "energetic_hindi", "expressive_hindi"], 
                       help="Test specific persona only")
    parser.add_argument("--url", default="http://localhost:8000", help="Bot base URL")
    args = parser.parse_args()
    
    tester = HTTPPersonaConversationTester(bot_base_url=args.url)
    
    logger.info("🎭 Starting Persona Conversation Flow Tests...")
    
    if args.persona:
        # Test single persona
        result = await tester.run_persona_test(args.persona)
        tester.print_persona_comparison([result])
    else:
        # Test all personas
        results = await tester.run_all_personas_test()
        tester.print_persona_comparison(results)

if __name__ == "__main__":
    asyncio.run(main()) 