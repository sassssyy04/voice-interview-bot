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

@dataclass 
class PersonaTestResult:
    """Test result for a single persona's complete conversation."""
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
                
                # Get raw ASR confidence data from the updated API
                raw_confidence_data = asr_data.get("raw_confidence_data", {})
                raw_asr_confidence = raw_confidence_data.get("model_confidence", asr_confidence)
                
                # Log detailed confidence information
                logger.info(f"  Raw ASR Data: Provider={raw_confidence_data.get('asr_provider', 'unknown')}, Source={raw_confidence_data.get('confidence_source', 'unknown')}")
                
                logger.info(f"\u2713 Turn completed in {latency_ms:.0f}ms")
                logger.info(f"  ASR: '{asr_text[:60]}...' (conf: {asr_confidence:.2f}, raw: {raw_asr_confidence:.2f})")
                logger.info(f"  Bot: '{bot_response[:60]}...'")
                logger.info(f"  Complete: {conversation_completed}")
                
                return ConversationTurn(
                    turn_number=0, step_name=step_name, audio_file=audio_file,
                    asr_text=asr_text, asr_confidence=raw_asr_confidence,  # Use raw ASR confidence
                    bot_response=bot_response, latency_ms=latency_ms,
                    success=True, extracted_entities=expected_entity,
                    current_field=current_field, candidate_profile=candidate_profile,
                    audio_variant=audio_variant
                )
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
            return bool(profile.get("pincode"))
        if step_name == "expected_salary":
            return profile.get("expected_salary") is not None
        if step_name == "has_two_wheeler":
            return profile.get("has_two_wheeler") is not None
        if step_name == "languages":
            langs = profile.get("languages") or []
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
        
        logger.info(f"\ud83c\udfad Starting {persona['name']} conversation test...")
        
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
            
            # Process each step in the persona's conversation flow
            for i, step_config in enumerate(persona["conversation_flow"], 1):
                logger.info(f"\n--- Step {i}/{len(persona['conversation_flow'])}: {step_config['step']} ---")
                # Resend mechanism: retry same audio up to 3 times if bot is stuck on same field
                max_retries = 3
                attempt = 0
                last_turn = None
                while attempt < max_retries:
                    turn_result = await self.send_audio_turn(
                        client, candidate_id,
                        step_config["audio_file"],
                        step_config["step"],
                        step_config.get("expected_entity", {}),
                        force_clean=False
                    )
                    last_turn = turn_result
                    # If HTTP failed, break and let outer logic handle
                    if not turn_result.success:
                        break
                    # If field got filled or conversation moved on, accept and proceed
                    if self._is_step_filled(step_config["step"], turn_result.candidate_profile):
                        break
                    # If bot's current_field is different from requested step, accept and move on
                    if turn_result.current_field and turn_result.current_field != step_config["step"]:
                        break
                    # If ASR produced no text or bot repeated same field, retry same audio
                    if (not turn_result.asr_text) or (turn_result.current_field == step_config["step"]):
                        attempt += 1
                        logger.warning(f"Bot might be stuck on '{step_config['step']}'. Retrying same audio (attempt {attempt}/{max_retries})...")
                        await asyncio.sleep(1)
                        continue
                    break

                turn_result = last_turn if last_turn else turn_result
                turns.append(turn_result)
                confidences.append(turn_result.asr_confidence)
                
                if turn_result.bot_response and ("job" in turn_result.bot_response.lower() or 
                                               "match" in turn_result.bot_response.lower()):
                    logger.info("\ud83c\udf89 Reached job matching!")
                    break
                
                if not turn_result.success:
                    logger.error(f"Failed on step {step_config['step']}, stopping conversation")
                    break
                
                # Small delay between turns
                await asyncio.sleep(2)
            
            end_time = datetime.now()
            
            # Calculate metrics
            total_latency = sum(turn.latency_ms for turn in turns)
            avg_latency = total_latency / len(turns) if turns else 0
            avg_confidence = sum(confidences) / len(confidences) if confidences else 0
            confidence_range = (min(confidences), max(confidences)) if confidences else (0, 0)
            
            final_turn = turns[-1] if turns else None
            conversation_completed = final_turn and ("job" in final_turn.bot_response.lower() 
                                                   or "match" in final_turn.bot_response.lower())
            
            return PersonaTestResult(
                persona_name=persona["name"],
                voice_id=persona["voice_id"],
                noise_levels=persona["noise_levels"],
                candidate_id=candidate_id,
                start_time=start_time,
                end_time=end_time,
                turns=turns,
                completed_steps=len(turns),
                total_steps=len(persona["conversation_flow"]),
                reached_job_matching=conversation_completed,
                final_bot_response=final_turn.bot_response if final_turn else "",
                conversation_completed=conversation_completed,
                total_latency_ms=total_latency,
                avg_turn_latency_ms=avg_latency,
                avg_confidence=avg_confidence,
                confidence_range=confidence_range
            )
    
    async def run_all_personas_test(self) -> List[PersonaTestResult]:
        """Run tests for all personas."""
        results = []
        
        for persona_key in self.personas.keys():
            try:
                result = await self.run_persona_test(persona_key)
                results.append(result)
                
                # Save individual result
                timestamp = int(time.time())
                filename = f"persona_{persona_key}_{timestamp}_with_preprocessing.json"
                filepath = self.results_dir / filename
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
                
                logger.info(f"Saved persona results to: {filepath}")
                
            except Exception as e:
                logger.error(f"Failed to test persona {persona_key}: {e}")
        
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
            
            if result.turns:
                print(f"   Confidence Variation:")
                for turn in result.turns:
                    print(f"     - {turn.step_name}: {turn.asr_confidence:.3f}")


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