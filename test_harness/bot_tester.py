"""Automated test harness for voice bot evaluation."""

import asyncio
import json
import time
import yaml
import base64
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import httpx
import websockets
from loguru import logger
from dataclasses import dataclass, asdict
from datetime import datetime
import numpy as np


@dataclass
class TestResult:
    """Single test execution result."""
    utterance_id: str
    audio_path: str
    expected_transcript: str
    expected_entities: Dict[str, Any]
    
    # Bot response
    actual_transcript: str
    actual_entities: Dict[str, Any]
    bot_response: str
    
    # Performance metrics
    asr_latency_ms: float
    nlu_latency_ms: float
    total_latency_ms: float
    turn_count: int
    completion_status: str
    
    # Accuracy metrics
    transcript_similarity: float
    entity_matches: Dict[str, bool]
    f1_scores: Dict[str, float]
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        return asdict(self)


@dataclass
class TestSession:
    """Complete test session results."""
    session_id: str
    start_time: datetime
    end_time: datetime
    test_results: List[TestResult]
    
    # Aggregate metrics
    overall_entity_f1: float
    per_slot_f1: Dict[str, float]
    completion_rate: float
    avg_latency_ms: float
    success_rate: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization."""
        result = asdict(self)
        result['start_time'] = self.start_time.isoformat()
        result['end_time'] = self.end_time.isoformat()
        return result


class BotTester:
    """Automated test harness for voice bot evaluation."""
    
    def __init__(self, bot_base_url: str = "http://localhost:8000", 
                 config_path: str = "utterances.yaml"):
        """Initialize the bot tester.
        
        Args:
            bot_base_url (str): Base URL for bot (HTTP)
            config_path (str): Path to test configuration
        """
        self.bot_base_url = bot_base_url
        self.config_path = config_path
        self.config = self._load_config()
        self.results_dir = Path("test_harness/results")
        self.results_dir.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> Dict:
        """Load test configuration from YAML file.
        
        Returns:
            Dict: Configuration dictionary
        """
        with open(self.config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    
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
    
    async def send_audio_to_bot(self, audio_path: str, candidate_id: str) -> Dict[str, Any]:
        """Send audio file to bot and collect response.
        
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
            logger.info(f"Connecting to bot at: {ws_url}")
            logger.info(f"Sending audio file: {audio_path} ({len(audio_data)} bytes)")
            
            # Connect to bot WebSocket
            async with websockets.connect(ws_url) as websocket:
                logger.info("WebSocket connected successfully")
                
                # Send audio data directly as bytes (the bot expects raw audio)
                audio_send_time = time.time()
                await websocket.send(audio_data)
                logger.info("Sent audio data to bot")
                
                # Wait for response
                logger.info("Waiting for bot response...")
                response = await asyncio.wait_for(
                    websocket.recv(), 
                    timeout=self.config["test_config"]["timeout_seconds"]
                )
                
                response_time = time.time()
                response_data = json.loads(response)
                
                # Calculate timing metrics
                total_latency = (response_time - start_time) * 1000
                processing_latency = (response_time - audio_send_time) * 1000
                
                logger.info(f"Received bot response in {total_latency:.0f}ms")
                logger.info(f"ASR Text: {response_data.get('asr', {}).get('text', 'N/A')}")
                
                return {
                    "success": True,
                    "response": response_data,
                    "total_latency_ms": total_latency,
                    "processing_latency_ms": processing_latency,
                    "asr_latency_ms": 0,  # Bot doesn't return this separately
                    "nlu_latency_ms": 0   # Bot doesn't return this separately
                }
                
        except asyncio.TimeoutError:
            logger.error(f"Timeout waiting for bot response: {audio_path}")
            return {"success": False, "error": "timeout"}
            
        except Exception as e:
            logger.error(f"Error sending audio to bot: {e}")
            return {"success": False, "error": str(e)}
    
    def calculate_transcript_similarity(self, expected: str, actual: str) -> float:
        """Calculate similarity between expected and actual transcripts.
        
        Args:
            expected (str): Expected transcript
            actual (str): Actual transcript
            
        Returns:
            float: Similarity score (0-1)
        """
        expected_words = set(expected.lower().split())
        actual_words = set(actual.lower().split())
        
        if not expected_words and not actual_words:
            return 1.0
        if not expected_words or not actual_words:
            return 0.0
            
        intersection = expected_words.intersection(actual_words)
        union = expected_words.union(actual_words)
        
        return len(intersection) / len(union) if union else 0.0
    
    def evaluate_entity_extraction(self, expected: Dict[str, Any], 
                                 actual: Dict[str, Any]) -> Tuple[Dict[str, bool], Dict[str, float]]:
        """Evaluate entity extraction accuracy.
        
        Args:
            expected (Dict[str, Any]): Expected entities
            actual (Dict[str, Any]): Actual entities
            
        Returns:
            Tuple[Dict[str, bool], Dict[str, float]]: Entity matches and F1 scores
        """
        entity_matches = {}
        f1_scores = {}
        
        # Get all entity keys
        all_keys = set(expected.keys()) | set(actual.keys())
        
        for key in all_keys:
            expected_val = expected.get(key)
            actual_val = actual.get(key)
            
            if expected_val is None and actual_val is None:
                entity_matches[key] = True
                f1_scores[key] = 1.0
            elif expected_val is None or actual_val is None:
                entity_matches[key] = False
                f1_scores[key] = 0.0
            else:
                # Handle different data types
                if isinstance(expected_val, list) and isinstance(actual_val, list):
                    # List comparison (e.g., languages)
                    expected_set = set(expected_val)
                    actual_set = set(actual_val)
                    
                    if expected_set == actual_set:
                        entity_matches[key] = True
                        f1_scores[key] = 1.0
                    else:
                        entity_matches[key] = False
                        # Calculate F1 for set comparison
                        if expected_set or actual_set:
                            precision = len(expected_set & actual_set) / len(actual_set) if actual_set else 0
                            recall = len(expected_set & actual_set) / len(expected_set) if expected_set else 0
                            f1_scores[key] = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
                        else:
                            f1_scores[key] = 1.0
                else:
                    # Direct comparison
                    match = str(expected_val).lower() == str(actual_val).lower()
                    entity_matches[key] = match
                    f1_scores[key] = 1.0 if match else 0.0
        
        return entity_matches, f1_scores
    
    async def run_single_test(self, utterance: Dict[str, Any], candidate_id: str) -> TestResult:
        """Run a single test case.
        
        Args:
            utterance (Dict[str, Any]): Test utterance configuration
            candidate_id (str): Candidate identifier
            
        Returns:
            TestResult: Test execution result
        """
        logger.info(f"Testing utterance: {utterance['id']}")
        
        # Find corresponding generated audio file
        audio_path = f"test_harness/generated_audio/{utterance['id']}_clean.wav"
        if utterance['noise_level'] != 'clean':
            noisy_path = audio_path.replace("_clean.wav", "_noisy.wav")
            if Path(noisy_path).exists():
                audio_path = noisy_path
        
        if not Path(audio_path).exists():
            logger.error(f"Audio file not found: {audio_path}")
            return TestResult(
                utterance_id=utterance["id"],
                audio_path=audio_path,
                expected_transcript=utterance["transcript"],
                expected_entities=utterance["entities"],
                actual_transcript="",
                actual_entities={},
                bot_response="FILE_NOT_FOUND",
                asr_latency_ms=0,
                nlu_latency_ms=0,
                total_latency_ms=0,
                turn_count=0,
                completion_status="failed",
                transcript_similarity=0.0,
                entity_matches={},
                f1_scores={}
            )
        
        # Send audio to bot
        bot_response = await self.send_audio_to_bot(audio_path, candidate_id)
        
        if not bot_response.get("success"):
            # Create failed test result
            return TestResult(
                utterance_id=utterance["id"],
                audio_path=audio_path,
                expected_transcript=utterance["transcript"],
                expected_entities=utterance["entities"],
                actual_transcript="",
                actual_entities={},
                bot_response=f"ERROR: {bot_response.get('error', 'unknown')}",
                asr_latency_ms=0,
                nlu_latency_ms=0,
                total_latency_ms=0,
                turn_count=0,
                completion_status="failed",
                transcript_similarity=0.0,
                entity_matches={},
                f1_scores={}
            )
        
        # Extract response data
        response_data = bot_response["response"]
        asr_data = response_data.get("asr", {})
        actual_transcript = asr_data.get("text", "")
        bot_text = response_data.get("text", "")
        
        # For now, we'll extract entities from the response text or use mock data
        # since the bot might not return structured entities in the WebSocket response
        actual_entities = {}  # Would need to parse from response or modify bot
        
        # Calculate metrics
        transcript_similarity = self.calculate_transcript_similarity(
            utterance["transcript"], actual_transcript
        )
        
        entity_matches, f1_scores = self.evaluate_entity_extraction(
            utterance["entities"], actual_entities
        )
        
        logger.info(f"✓ Test completed: {utterance['id']} - Transcript: {actual_transcript[:50]}...")
        
        return TestResult(
            utterance_id=utterance["id"],
            audio_path=audio_path,
            expected_transcript=utterance["transcript"],
            expected_entities=utterance["entities"],
            actual_transcript=actual_transcript,
            actual_entities=actual_entities,
            bot_response=bot_text,
            asr_latency_ms=bot_response.get("asr_latency_ms", 0),
            nlu_latency_ms=bot_response.get("nlu_latency_ms", 0),
            total_latency_ms=bot_response.get("total_latency_ms", 0),
            turn_count=1,
            completion_status="completed",
            transcript_similarity=transcript_similarity,
            entity_matches=entity_matches,
            f1_scores=f1_scores
        )
    
    def calculate_aggregate_metrics(self, test_results: List[TestResult]) -> Dict[str, float]:
        """Calculate aggregate metrics across all tests.
        
        Args:
            test_results (List[TestResult]): Individual test results
            
        Returns:
            Dict[str, float]: Aggregate metrics
        """
        if not test_results:
            return {}
        
        # Collect all F1 scores by entity type
        all_f1_scores = {}
        for result in test_results:
            for entity, f1 in result.f1_scores.items():
                if entity not in all_f1_scores:
                    all_f1_scores[entity] = []
                all_f1_scores[entity].append(f1)
        
        # Calculate per-slot F1 (macro average)
        per_slot_f1 = {}
        for entity, scores in all_f1_scores.items():
            per_slot_f1[entity] = np.mean(scores) if scores else 0.0
        
        # Overall entity F1 (macro average across slots)
        overall_entity_f1 = np.mean(list(per_slot_f1.values())) if per_slot_f1 else 0.0
        
        # Completion rate (successful tests)
        successful_tests = [r for r in test_results if r.completion_status == "completed"]
        completion_rate = len(successful_tests) / len(test_results)
        
        # Average latency
        latencies = [r.total_latency_ms for r in successful_tests]
        avg_latency_ms = np.mean(latencies) if latencies else 0.0
        
        # Success rate (based on transcript similarity for now)
        transcript_similarities = [r.transcript_similarity for r in successful_tests]
        success_count = sum(1 for sim in transcript_similarities if sim >= 0.5)
        success_rate = success_count / len(test_results)
        
        return {
            "overall_entity_f1": overall_entity_f1,
            "per_slot_f1": per_slot_f1,
            "completion_rate": completion_rate,
            "avg_latency_ms": avg_latency_ms,
            "success_rate": success_rate
        }
    
    async def run_test_suite(self, utterance_ids: Optional[List[str]] = None) -> TestSession:
        """Run complete test suite.
        
        Args:
            utterance_ids (Optional[List[str]]): Specific utterances to test
            
        Returns:
            TestSession: Complete test session results
        """
        start_time = datetime.now()
        session_id = f"test_session_{int(time.time())}"
        
        logger.info(f"Starting test session: {session_id}")
        logger.info(f"Bot URL: {self.bot_base_url}")
        
        # Filter utterances if specific IDs provided
        utterances = self.config["utterances"]
        if utterance_ids:
            utterances = [u for u in utterances if u["id"] in utterance_ids]
        
        logger.info(f"Testing {len(utterances)} utterances")
        
        # Run individual tests
        test_results = []
        for i, utterance in enumerate(utterances, 1):
            try:
                logger.info(f"Test {i}/{len(utterances)}: {utterance['id']}")
                result = await self.run_single_test(utterance, f"test_{utterance['id']}")
                test_results.append(result)
                
                # Brief pause between tests
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"Error testing {utterance['id']}: {e}")
        
        # Calculate aggregate metrics
        metrics = self.calculate_aggregate_metrics(test_results)
        
        end_time = datetime.now()
        
        test_session = TestSession(
            session_id=session_id,
            start_time=start_time,
            end_time=end_time,
            test_results=test_results,
            overall_entity_f1=metrics.get("overall_entity_f1", 0.0),
            per_slot_f1=metrics.get("per_slot_f1", {}),
            completion_rate=metrics.get("completion_rate", 0.0),
            avg_latency_ms=metrics.get("avg_latency_ms", 0.0),
            success_rate=metrics.get("success_rate", 0.0)
        )
        
        # Save results
        self.save_test_results(test_session)
        
        return test_session
    
    def save_test_results(self, test_session: TestSession) -> str:
        """Save test results to file.
        
        Args:
            test_session (TestSession): Test session to save
            
        Returns:
            str: Path to saved results file
        """
        results_file = self.results_dir / f"{test_session.session_id}_results.json"
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(test_session.to_dict(), f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved test results to: {results_file}")
        return str(results_file)
    
    def print_metrics_report(self, test_session: TestSession):
        """Print formatted metrics report.
        
        Args:
            test_session (TestSession): Test session results
        """
        print("\n" + "="*60)
        print(f"HINGLISH VOICE BOT TEST RESULTS")
        print(f"Session ID: {test_session.session_id}")
        print(f"Duration: {(test_session.end_time - test_session.start_time).total_seconds():.1f}s")
        print("="*60)
        
        print(f"\nOVERALL METRICS:")
        print(f"  Completion Rate:     {test_session.completion_rate:.1%}")
        print(f"  Success Rate:        {test_session.success_rate:.1%}")
        print(f"  Average Latency:     {test_session.avg_latency_ms:.0f}ms")
        
        print(f"\nSAMPLE RESULTS:")
        for i, result in enumerate(test_session.test_results[:5]):  # Show first 5
            print(f"  {result.utterance_id}:")
            print(f"    Expected: {result.expected_transcript[:60]}...")
            print(f"    Actual:   {result.actual_transcript[:60]}...")
            print(f"    Similarity: {result.transcript_similarity:.3f}")
            print()
        
        print("="*60)


async def main():
    """Run the complete test suite."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Hinglish Voice Bot")
    parser.add_argument("--utterances", nargs="+", help="Specific utterance IDs to test")
    parser.add_argument("--url", default="http://localhost:8000", help="Bot base URL")
    args = parser.parse_args()
    
    tester = BotTester(bot_base_url=args.url)
    
    logger.info("Starting automated bot testing...")
    test_session = await tester.run_test_suite(utterance_ids=args.utterances)
    
    # Print results
    tester.print_metrics_report(test_session)


if __name__ == "__main__":
    asyncio.run(main()) 