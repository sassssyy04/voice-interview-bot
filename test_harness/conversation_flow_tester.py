"""Multi-turn conversation flow tester for the Hinglish Voice Bot."""

import asyncio
import json
import time
import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional
import httpx
import websockets
from loguru import logger
from dataclasses import dataclass, asdict
from datetime import datetime


@dataclass
class ConversationTurn:
    """Single conversation turn."""
    turn_id: str
    step_name: str
    expected_question: str
    audio_file: str
    expected_entity: Dict[str, Any]
    
    # Response data
    bot_response: str = ""
    asr_text: str = ""
    asr_confidence: float = 0.0
    latency_ms: float = 0.0
    success: bool = False
    extracted_entity: Dict[str, Any] = None


@dataclass
class ConversationTestResult:
    """Complete conversation test result."""
    session_id: str
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
    entity_extraction_accuracy: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class ConversationFlowTester:
    """Tests the complete conversation flow from greeting to job matching."""
    
    def __init__(self, bot_base_url: str = "http://localhost:8000"):
        """Initialize the conversation flow tester.
        
        Args:
            bot_base_url (str): Base URL for bot (HTTP)
        """
        self.bot_base_url = bot_base_url
        self.results_dir = Path("test_harness/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Define the conversation flow mapping audio files to steps
        # Based on the actual generated audio files we have
        self.conversation_mapping = [
            {
                "step": "pincode", 
                "question": "Aap kahan rehte hain? Apna pincode ya area batayiye।",
                "audio_files": ["pin_001", "pin_002", "pin_003"],  # Delhi, Mumbai, Bangalore
                "expected_entity": {"pincode": "110001"}
            },
            {
                "step": "expected_salary",
                "question": "Aapko kitni salary chahiye har mahine? Rupees mein batayiye।", 
                "audio_files": ["sal_001", "sal_002", "sal_003"],  # 15000, 25000, etc
                "expected_entity": {"expected_salary": 15000}
            },
            {
                "step": "has_two_wheeler",
                "question": "Kya aapke paas bike ya scooter hai?",
                "audio_files": ["vehicle_001", "vehicle_002"],  # "haan hai", "nahi hai"
                "expected_entity": {"has_two_wheeler": True}
            },
            {
                "step": "languages",
                "question": "Aap kaunsi languages bol sakte hain? Hindi, English ya koi aur?",
                "audio_files": ["lang_001", "lang_002"],  # Hindi English, Hindi Tamil
                "expected_entity": {"languages": ["hindi", "english"]}
            },
            {
                "step": "availability_date", 
                "question": "Aap kab se kaam shuru kar sakte hain? Aaj, kal ya koi aur din?",
                "audio_files": ["avail_001", "avail_002"],  # "kal se", "next week"
                "expected_entity": {"availability_date": "tomorrow"}
            },
            {
                "step": "preferred_shift",
                "question": "Aap kaunse time pe kaam karna chahte hain? Morning, evening ya night?",
                "audio_files": ["shift_001", "shift_002"],  # "morning", "any shift"
                "expected_entity": {"preferred_shift": "morning"}
            },
            {
                "step": "total_experience_months",
                "question": "Aapko kitna kaam ka experience hai? Kitne saal ya mahine?",
                "audio_files": ["exp_001", "exp_002"],  # "2 saal", "6 months"
                "expected_entity": {"total_experience_months": 24}
            }
        ]
    
    async def start_conversation_session(self) -> str:
        """Start a new conversation session and get candidate_id.
        
        Returns:
            str: Candidate ID for the session
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(f"{self.bot_base_url}/api/v1/conversation/start")
                if response.status_code == 200:
                    data = response.json()
                    candidate_id = data.get("candidate_id")
                    logger.info(f"Started conversation session: {candidate_id}")
                    return candidate_id
                else:
                    logger.error(f"Failed to start conversation: {response.status_code}")
                    return f"test_session_{int(time.time())}"
        except Exception as e:
            logger.error(f"Error starting conversation: {e}")
            return f"test_session_{int(time.time())}"
    
    async def send_audio_turn(self, audio_path: str, candidate_id: str) -> Dict[str, Any]:
        """Send a single audio turn to the bot.
        
        Args:
            audio_path (str): Path to audio file
            candidate_id (str): Candidate identifier
            
        Returns:
            Dict[str, Any]: Bot response with timing information
        """
        start_time = time.time()
        
        try:
            # Read audio file as bytes
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            
            # Construct WebSocket URL with candidate_id
            ws_url = self.bot_base_url.replace("http://", "ws://") + f"/api/v1/ws/{candidate_id}"
            logger.info(f"Sending turn: {audio_path} ({len(audio_data)} bytes)")
            
            # Connect to bot WebSocket
            async with websockets.connect(ws_url) as websocket:
                # Send audio data directly as bytes
                await websocket.send(audio_data)
                
                # Wait for response
                response = await asyncio.wait_for(websocket.recv(), timeout=45)
                
                response_time = time.time()
                response_data = json.loads(response)
                
                # Calculate timing metrics
                total_latency = (response_time - start_time) * 1000
                
                asr_data = response_data.get("asr", {})
                asr_text = asr_data.get("text", "")
                asr_confidence = asr_data.get("confidence", 0.0)
                bot_text = response_data.get("text", "")
                conversation_complete = response_data.get("conversation_complete", False)
                
                logger.info(f"✓ Turn completed in {total_latency:.0f}ms")
                logger.info(f"  ASR: '{asr_text}' (conf: {asr_confidence:.2f})")
                logger.info(f"  Bot: '{bot_text[:80]}...'")
                logger.info(f"  Complete: {conversation_complete}")
                
                return {
                    "success": True,
                    "asr_text": asr_text,
                    "asr_confidence": asr_confidence,
                    "bot_response": bot_text,
                    "conversation_complete": conversation_complete,
                    "latency_ms": total_latency
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout on turn: {audio_path}")
            return {"success": False, "error": "timeout"}
        except Exception as e:
            logger.error(f"Error on turn: {e}")
            return {"success": False, "error": str(e)}
    
    async def run_full_conversation_test(self, test_name: str = "default") -> ConversationTestResult:
        """Run a complete conversation flow test.
        
        Args:
            test_name (str): Name identifier for this test
            
        Returns:
            ConversationTestResult: Complete conversation test results
        """
        start_time = datetime.now()
        session_id = f"conversation_test_{test_name}_{int(time.time())}"
        
        logger.info(f"🎭 Starting full conversation test: {session_id}")
        
        # Start conversation session
        candidate_id = await self.start_conversation_session()
        
        turns = []
        completed_steps = 0
        conversation_complete = False
        final_bot_response = ""
        reached_job_matching = False
        
        # Run each conversation step
        for i, step_config in enumerate(self.conversation_mapping):
            step_name = step_config["step"]
            audio_files = step_config["audio_files"]
            expected_entity = step_config["expected_entity"]
            
            logger.info(f"\n--- Step {i+1}/{len(self.conversation_mapping)}: {step_name} ---")
            
            # Use first available audio file for this step
            audio_file = None
            for audio_id in audio_files:
                audio_path = f"test_harness/generated_audio/{audio_id}_clean.wav"
                if Path(audio_path).exists():
                    audio_file = audio_path
                    break
            
            if not audio_file:
                logger.error(f"No audio file found for step {step_name}")
                # Create failed turn
                turn = ConversationTurn(
                    turn_id=f"turn_{i+1}",
                    step_name=step_name,
                    expected_question=step_config["question"],
                    audio_file="MISSING",
                    expected_entity=expected_entity,
                    success=False
                )
                turns.append(turn)
                break
            
            # Send the audio turn
            response = await self.send_audio_turn(audio_file, candidate_id)
            
            # Create turn result
            turn = ConversationTurn(
                turn_id=f"turn_{i+1}",
                step_name=step_name,
                expected_question=step_config["question"],
                audio_file=audio_file,
                expected_entity=expected_entity,
                bot_response=response.get("bot_response", ""),
                asr_text=response.get("asr_text", ""),
                asr_confidence=response.get("asr_confidence", 0.0),
                latency_ms=response.get("latency_ms", 0.0),
                success=response.get("success", False)
            )
            turns.append(turn)
            
            if turn.success:
                completed_steps += 1
                final_bot_response = turn.bot_response
                
                # Check if conversation is complete
                if response.get("conversation_complete", False):
                    conversation_complete = True
                    # Check if final response mentions jobs or matching
                    if any(keyword in final_bot_response.lower() for keyword in 
                           ["job", "match", "position", "opportunity", "sms", "message"]):
                        reached_job_matching = True
                    break
            else:
                logger.error(f"Failed on step {step_name}, stopping conversation")
                break
            
            # Brief pause between turns
            await asyncio.sleep(2)
        
        end_time = datetime.now()
        
        # Calculate metrics
        successful_turns = [t for t in turns if t.success]
        total_latency = sum(turn.latency_ms for turn in successful_turns)
        avg_latency = total_latency / len(successful_turns) if successful_turns else 0
        
        result = ConversationTestResult(
            session_id=session_id,
            candidate_id=candidate_id,
            start_time=start_time,
            end_time=end_time,
            turns=turns,
            completed_steps=completed_steps,
            total_steps=len(self.conversation_mapping),
            reached_job_matching=reached_job_matching,
            final_bot_response=final_bot_response,
            conversation_completed=conversation_complete,
            total_latency_ms=total_latency,
            avg_turn_latency_ms=avg_latency,
            entity_extraction_accuracy=0.0  # TODO: Implement entity accuracy calculation
        )
        
        # Save results
        self.save_test_results(result)
        
        return result
    
    def save_test_results(self, result: ConversationTestResult) -> str:
        """Save conversation test results to file.
        
        Args:
            result (ConversationTestResult): Test results to save
            
        Returns:
            str: Path to saved results file
        """
        results_file = self.results_dir / f"{result.session_id}_conversation.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved conversation results to: {results_file}")
        return str(results_file)
    
    def print_conversation_report(self, result: ConversationTestResult):
        """Print formatted conversation test report.
        
        Args:
            result (ConversationTestResult): Conversation test results
        """
        print("\n" + "="*70)
        print(f"🎭 HINGLISH VOICE BOT - FULL CONVERSATION TEST")
        print(f"Session ID: {result.session_id}")
        print(f"Duration: {(result.end_time - result.start_time).total_seconds():.1f}s")
        print("="*70)
        
        print(f"\n📊 CONVERSATION FLOW RESULTS:")
        print(f"  Steps Completed:     {result.completed_steps}/{result.total_steps}")
        print(f"  Conversation Complete: {'✅ YES' if result.conversation_completed else '❌ NO'}")
        print(f"  Reached Job Matching: {'✅ YES' if result.reached_job_matching else '❌ NO'}")
        print(f"  Average Turn Latency: {result.avg_turn_latency_ms:.0f}ms")
        
        print(f"\n🔄 TURN-BY-TURN BREAKDOWN:")
        for i, turn in enumerate(result.turns, 1):
            status = "✅" if turn.success else "❌"
            asr_preview = turn.asr_text[:30] + "..." if len(turn.asr_text) > 30 else turn.asr_text
            bot_preview = turn.bot_response[:50] + "..." if len(turn.bot_response) > 50 else turn.bot_response
            
            print(f"  {status} Turn {i}: {turn.step_name}")
            print(f"     ASR: '{asr_preview}' (conf: {turn.asr_confidence:.2f})")
            print(f"     Bot: '{bot_preview}'")
            print(f"     Time: {turn.latency_ms:.0f}ms")
            print()
        
        if result.reached_job_matching:
            print(f"🎉 SUCCESS: Bot completed full conversation flow!")
            print(f"📄 Final Response: {result.final_bot_response}")
        else:
            print(f"⚠️  INCOMPLETE: Conversation stopped at step {result.completed_steps}")
        
        print("="*70)


async def main():
    """Run conversation flow tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hinglish Voice Bot Conversation Flow")
    parser.add_argument("--test-name", default="full_flow", help="Name for this test run")
    parser.add_argument("--url", default="http://localhost:8000", help="Bot base URL")
    args = parser.parse_args()
    
    tester = ConversationFlowTester(bot_base_url=args.url)
    
    logger.info("🎭 Starting full conversation flow test...")
    result = await tester.run_full_conversation_test(test_name=args.test_name)
    
    # Print results
    tester.print_conversation_report(result)


if __name__ == "__main__":
    asyncio.run(main()) 