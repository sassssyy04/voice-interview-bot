import base64
import json
from typing import Dict, Any, List
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, File, UploadFile
from fastapi.responses import JSONResponse
from app.services.conversation import ConversationOrchestrator
from app.core.logger import logger

router = APIRouter()

# Global conversation orchestrator
orchestrator = ConversationOrchestrator()

# Active WebSocket connections
active_connections: Dict[str, WebSocket] = {}


@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy", "service": "hinglish-voice-bot"}


@router.post("/conversation/start")
async def start_conversation():
    """Start a new voice conversation session."""
    try:
        candidate_id, audio_data = await orchestrator.start_conversation()
        
        # Convert audio to base64 for web transmission
        audio_base64 = base64.b64encode(audio_data).decode('utf-8')
        
        return {
            "candidate_id": candidate_id,
            "audio_data": audio_base64,
            "audio_format": "wav",
            "message": "Conversation started successfully"
        }
    
    except Exception as e:
        logger.error(f"Failed to start conversation: {e}")
        # Fallback to demo mode if voice fails
        return {
            "candidate_id": "demo_mode",
            "audio_data": "",
            "audio_format": "text", 
            "message": f"Voice system error: {str(e)}. Enable Google Cloud APIs to use voice."
        }


@router.post("/conversation/{candidate_id}/turn")
async def process_voice_turn(candidate_id: str, audio_file: UploadFile = File(...)):
    """Process a voice turn in the conversation."""
    try:
        # Read audio data
        audio_data = await audio_file.read()

        # Ignore too-short audio to prevent premature "didn't understand" replies
        MIN_AUDIO_BYTES = 4096
        if len(audio_data) < MIN_AUDIO_BYTES:
            try:
                prompt_text = "Kripya button ko dabaye rakhen aur bolein."
                if orchestrator.tts_service:
                    tts_audio = await orchestrator.tts_service.synthesize_speech(prompt_text)
                    response_audio_base64 = base64.b64encode(tts_audio).decode('utf-8')
                else:
                    response_audio_base64 = ""
                metrics = orchestrator.get_conversation_metrics(candidate_id)
                return {
                    "candidate_id": candidate_id,
                    "text": prompt_text,
                    "audio_data": response_audio_base64,
                    "audio_format": "wav" if response_audio_base64 else "text",
                    "conversation_complete": False,
                    "metrics": metrics
                }
            except Exception:
                # If any issue generating TTS, fall back to normal processing
                pass
         
        # Process the turn
        response_text, response_audio, conversation_complete = await orchestrator.process_turn(
            candidate_id, audio_data
        )
        
        # Convert response audio to base64
        response_audio_base64 = base64.b64encode(response_audio).decode('utf-8')
        
        # Get conversation metrics
        metrics = orchestrator.get_conversation_metrics(candidate_id)
        
        return {
            "candidate_id": candidate_id,
            "text": response_text,
            "audio_data": response_audio_base64,
            "audio_format": "wav",
            "conversation_complete": conversation_complete,
            "metrics": metrics
        }
    
    except Exception as e:
        logger.error(f"Failed to process turn: {e}")
        raise HTTPException(status_code=500, detail="Failed to process voice turn")


@router.get("/conversation/{candidate_id}/status")
async def get_conversation_status(candidate_id: str):
    """Get current conversation status and metrics."""
    try:
        if candidate_id not in orchestrator.conversation_states:
            raise HTTPException(status_code=404, detail="Conversation not found")
        
        state = orchestrator.conversation_states[candidate_id]
        candidate = orchestrator.candidates[candidate_id]
        metrics = orchestrator.get_conversation_metrics(candidate_id)
        
        return {
            "candidate_id": candidate_id,
            "current_field": state.current_field,
            "completion_rate": state.completion_rate,
            "fields_completed": state.fields_completed,
            "conversation_complete": candidate.conversation_completed,
            "metrics": metrics,
            "candidate_profile": candidate.dict()
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get conversation status: {e}")
        raise HTTPException(status_code=500, detail="Failed to get conversation status")


@router.get("/conversation/{candidate_id}/matches")
async def get_job_matches(candidate_id: str):
    """Get job matches for a candidate."""
    try:
        if candidate_id not in orchestrator.candidates:
            raise HTTPException(status_code=404, detail="Candidate not found")
        
        candidate = orchestrator.candidates[candidate_id]
        
        if not candidate.conversation_completed:
            raise HTTPException(status_code=400, detail="Conversation not complete")
        
        # Generate job matches
        matches = await orchestrator.job_matching_service.find_job_matches(candidate)
        
        return {
            "candidate_id": candidate_id,
            "matches": [
                {
                    "job": match.job.dict(),
                    "match_score": match.match_score,
                    "rationale": match.rationale,
                    "strengths": match.strengths,
                    "concerns": match.concerns,
                    "score_breakdown": {
                        "location": match.location_score,
                        "salary": match.salary_score,
                        "shift": match.shift_score,
                        "language": match.language_score,
                        "vehicle": match.vehicle_score,
                        "experience": match.experience_score
                    }
                }
                for match in matches.top_matches
            ],
            "total_jobs_considered": matches.total_jobs_considered,
            "matching_criteria": matches.matching_criteria_used
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Failed to get job matches: {e}")
        raise HTTPException(status_code=500, detail="Failed to get job matches")


@router.get("/metrics/dashboard")
async def get_dashboard_metrics():
    """Get aggregated metrics for dashboard."""
    try:
        total_conversations = len(orchestrator.candidates)
        completed_conversations = sum(1 for c in orchestrator.candidates.values() if c.conversation_completed)
        
        if total_conversations == 0:
            return {
                "total_conversations": 0,
                "completion_rate": 0.0,
                "avg_turns": 0.0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0
            }
        
        # Calculate aggregated metrics
        all_metrics = [orchestrator.get_conversation_metrics(cid) for cid in orchestrator.candidates.keys()]
        valid_metrics = [m for m in all_metrics if m]
        
        if not valid_metrics:
            return {
                "total_conversations": total_conversations,
                "completion_rate": 0.0,
                "avg_turns": 0.0,
                "avg_latency_ms": 0.0,
                "avg_confidence": 0.0
            }
        
        avg_turns = sum(m["total_turns"] for m in valid_metrics) / len(valid_metrics)
        avg_latency = sum(m["avg_latency_ms"] for m in valid_metrics) / len(valid_metrics)
        avg_confidence = sum(m["avg_confidence"] for m in valid_metrics) / len(valid_metrics)
        
        return {
            "total_conversations": total_conversations,
            "completed_conversations": completed_conversations,
            "completion_rate": completed_conversations / total_conversations,
            "avg_turns": avg_turns,
            "avg_latency_ms": avg_latency,
            "avg_confidence": avg_confidence,
            "p50_latency_ms": sum(m["p50_latency_ms"] for m in valid_metrics) / len(valid_metrics),
            "p95_latency_ms": sum(m["p95_latency_ms"] for m in valid_metrics) / len(valid_metrics)
        }
    
    except Exception as e:
        logger.error(f"Failed to get dashboard metrics: {e}")
        raise HTTPException(status_code=500, detail="Failed to get dashboard metrics")


@router.websocket("/ws/{candidate_id}")
async def websocket_endpoint(websocket: WebSocket, candidate_id: str):
    """WebSocket endpoint for real-time voice conversation."""
    await websocket.accept()
    active_connections[candidate_id] = websocket
    
    try:
        while True:
            # Receive audio data from client
            data = await websocket.receive_bytes()
            
            # Process the voice turn
            response_text, response_audio, conversation_complete = await orchestrator.process_turn(
                candidate_id, data
            )
            
            # Send response back to client
            response_data = {
                "text": response_text,
                "audio_data": base64.b64encode(response_audio).decode('utf-8'),
                "conversation_complete": conversation_complete
            }
            
            await websocket.send_text(json.dumps(response_data))
            
            if conversation_complete:
                break
                
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for candidate {candidate_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        if candidate_id in active_connections:
            del active_connections[candidate_id]


@router.get("/test/noise")
async def test_with_noise():
    """Test endpoint to simulate noisy environment."""
    return {
        "message": "Play background scooter noise during demo",
        "instructions": "Add 10s traffic sound loop for noise testing",
        "noise_simulation": "enabled"
    }

@router.post("/demo/job-match")
async def demo_job_match():
    """Demo endpoint to show job matching functionality."""
    from app.models.candidate import Candidate, ShiftPreference, LanguageSkill
    
    # Create sample candidate
    sample_candidate = Candidate(
        candidate_id="demo_001",
        pincode="110001",
        locality="Connaught Place",
        availability_date="immediately",
        preferred_shift=ShiftPreference.MORNING,
        expected_salary=18000,
        languages=[LanguageSkill.HINDI, LanguageSkill.ENGLISH],
        has_two_wheeler=True,
        total_experience_months=6,
        conversation_completed=True
    )
    
    # Get job matches
    matches = await orchestrator.job_matching_service.find_job_matches(sample_candidate)
    
    return {
        "candidate": {
            "pincode": sample_candidate.pincode,
            "expected_salary": sample_candidate.expected_salary,
            "preferred_shift": sample_candidate.preferred_shift.value,
            "languages": [l.value for l in sample_candidate.languages],
            "has_two_wheeler": sample_candidate.has_two_wheeler,
            "experience_months": sample_candidate.total_experience_months
        },
        "matches": [
            {
                "job": {
                    "title": match.job.title,
                    "company": match.job.company,
                    "locality": match.job.locality,
                    "salary_min": match.job.salary_min,
                    "salary_max": match.job.salary_max,
                    "contact_number": match.job.contact_number,
                    "description": match.job.description
                },
                "match_score": round(match.match_score * 100),
                "rationale": match.rationale,
                "strengths": match.strengths,
                "concerns": match.concerns,
                "score_breakdown": {
                    "location": round(match.location_score * 100),
                    "salary": round(match.salary_score * 100),
                    "shift": round(match.shift_score * 100),
                    "language": round(match.language_score * 100),
                    "vehicle": round(match.vehicle_score * 100),
                    "experience": round(match.experience_score * 100)
                }
            }
            for match in matches.top_matches
        ],
        "total_jobs_considered": matches.total_jobs_considered
    } 