from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse
from typing import Dict, Any
import json
import asyncio
import logging

from app.services.whisper_chatterbox_service import whisper_chatterbox_service
from app.utils.supabase_auth import verify_supabase_token

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/voice-whisper",
    tags=["voice-whisper"],
)

logger.info("✅ [VOICE WHISPER] Router created with prefix /api/voice-whisper")


@router.get("/events/{user_id}")
async def voice_events_stream_whisper(
    user_id: str, 
    is_audio: str = "true",
    user_info: Dict[str, Any] = Depends(verify_supabase_token)
):
    """
    Server-Sent Events (SSE) endpoint for voice assistant using Whisper + Chatterbox.
    
    Args:
        user_id: User identifier
        is_audio: Whether to use audio mode ("true" or "false")
        user_info: Authenticated user information
    """
    logger.info(f"🎯 [VOICE WHISPER] GET /events/{user_id} - ENTRY POINT")
    logger.info(f"Voice client {user_id} connecting via Whisper+Chatterbox SSE, audio mode: {is_audio}")
    logger.info(f"✅ Authenticated user: {user_info.get('email', 'unknown')}")
    
    # Authenticate UserID
    if user_info.get('sub') != user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch")
    
    try:
        # Create voice session with Whisper + Chatterbox
        is_audio_mode = is_audio.lower() == "true"
        session_data = await whisper_chatterbox_service.create_session(
            user_id=user_id, 
            is_audio=is_audio_mode
        )
        session_id = session_data["session_id"]
        
        async def event_generator():
            try:
                # Send session created event
                yield f"data: {json.dumps({'type': 'session_created', 'session_id': session_id})}\n\n"
                
                # Send service info
                yield f"data: {json.dumps({'type': 'service_info', 'service': 'whisper_chatterbox', 'models': {'asr': 'whisper', 'tts': 'chatterbox'}})}\n\n"
                
                # Keep connection alive with heartbeats
                heartbeat_interval = 30  # seconds
                
                while True:
                    # Check if session still exists
                    current_session = await whisper_chatterbox_service.get_session(session_id)
                    if not current_session:
                        logger.info(f"Session {session_id} no longer exists, closing SSE")
                        break
                    
                    # Send heartbeat
                    heartbeat_event = {
                        "type": "heartbeat",
                        "timestamp": current_session['last_active'],
                        "session_id": session_id
                    }
                    yield f"data: {json.dumps(heartbeat_event)}\n\n"
                    
                    # Wait for next heartbeat
                    await asyncio.sleep(heartbeat_interval)
                    
            except Exception as e:
                logger.error(f"Error in Whisper+Chatterbox SSE stream: {e}")
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
            finally:
                # Clean up session
                await whisper_chatterbox_service.close_session(session_id)
                logger.info(f"Voice client {user_id} disconnected from Whisper+Chatterbox SSE")
        
        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except Exception as e:
        logger.error(f"Error creating Whisper+Chatterbox session for user {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to create voice session: {str(e)}")


@router.post("/send/{session_id}")
async def send_voice_message_whisper(
    session_id: str, 
    request: Request,
    user_info: Dict[str, Any] = Depends(verify_supabase_token)
):
    """
    HTTP endpoint for sending messages to the Whisper + Chatterbox voice assistant.
    
    Args:
        session_id: Session identifier
        request: HTTP request containing the message
        user_info: Authenticated user information
    """
    logger.info(f"🔍 [VOICE WHISPER] POST /send/{session_id} - START")
    logger.info(f"✅ Authenticated user sending Whisper+Chatterbox message: {user_info.get('email', 'unknown')}")
    
    try:
        # Parse the message
        logger.info(f"🔍 [VOICE WHISPER] Parsing request JSON...")
        message = await request.json()
        mime_type = message.get("mime_type")
        data = message.get("data")
        logger.info(f"🔍 [VOICE WHISPER] Parsed: mime_type={mime_type}, data_length={len(data) if data else 0}")
        
        if not mime_type or not data:
            logger.info(f"❌ [VOICE WHISPER] Missing mime_type or data")
            raise HTTPException(status_code=400, detail="Missing mime_type or data")
        
        # Verify session exists and belongs to user
        logger.info(f"🔍 [VOICE WHISPER] Verifying session {session_id}...")
        session_data = await whisper_chatterbox_service.get_session(session_id)
        if not session_data:
            logger.info(f"❌ [VOICE WHISPER] Session {session_id} not found")
            raise HTTPException(status_code=404, detail="Session not found")
        
        logger.info(f"✅ [VOICE WHISPER] Session {session_id} found")
        if session_data.get("user_id") != user_info.get('sub'):
            logger.info(f"❌ [VOICE WHISPER] Session does not belong to user")
            raise HTTPException(status_code=403, detail="Session does not belong to user")
        
        # Stream the response back
        logger.info(f"🔍 [VOICE WHISPER] Starting response stream...")
        async def response_generator():
            try:
                logger.info(f"🔍 [VOICE WHISPER] Calling whisper_chatterbox_service.send_message...")
                async for event_data in whisper_chatterbox_service.send_message(session_id, data, mime_type):
                    logger.info(f"🔍 [VOICE WHISPER] Got event: {event_data.get('type', 'unknown')}")
                    yield f"data: {json.dumps(event_data)}\n\n"
                    await asyncio.sleep(0.01)
                logger.info(f"✅ [VOICE WHISPER] Finished processing events")
            except Exception as e:
                logger.error(f"❌ [VOICE WHISPER] Error in response generator: {e}")
                import traceback
                traceback.print_exc()
                yield f"data: {json.dumps({'type': 'error', 'message': str(e)})}\n\n"
        
        if mime_type == "text/plain":
            logger.info(f"[CLIENT TO WHISPER VOICE] Session {session_id}: {data}")
        elif mime_type == "audio/pcm":
            logger.info(f"[CLIENT TO WHISPER VOICE] Session {session_id}: audio/pcm: {len(data)} chars (base64)")
        
        return StreamingResponse(
            response_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
                "Access-Control-Allow-Headers": "Cache-Control"
            }
        )
        
    except json.JSONDecodeError:
        raise HTTPException(status_code=400, detail="Invalid JSON format")
    except Exception as e:
        logger.error(f"Error sending Whisper+Chatterbox message for session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to send message: {str(e)}")


@router.get("/sessions/{user_id}")
async def get_user_whisper_voice_sessions(
    user_id: str,
    user_info: Dict[str, Any] = Depends(verify_supabase_token)
):
    """
    Get all active voice sessions for a user.
    
    Args:
        user_id: User identifier
        user_info: Authenticated user information
    """
    if user_info.get('sub') != user_id:
        raise HTTPException(status_code=403, detail="User ID mismatch")
    
    try:
        active_sessions = await whisper_chatterbox_service.list_active_sessions()
        user_sessions = []
        
        for session_id in active_sessions:
            session_data = await whisper_chatterbox_service.get_session(session_id)
            if session_data and session_data.get("user_id") == user_id:
                user_sessions.append(session_data)
        
        return {"sessions": user_sessions}
        
    except Exception as e:
        logger.error(f"Error getting user sessions: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get sessions: {str(e)}")


@router.delete("/sessions/{session_id}")
async def cleanup_whisper_voice_session(
    session_id: str,
    user_info: Dict[str, Any] = Depends(verify_supabase_token)
):
    """
    Clean up a specific voice session.
    
    Args:
        session_id: Session identifier
        user_info: Authenticated user information
    """
    try:
        # Verify session belongs to user
        session_data = await whisper_chatterbox_service.get_session(session_id)
        if not session_data:
            raise HTTPException(status_code=404, detail="Session not found")
        
        if session_data.get("user_id") != user_info.get('sub'):
            raise HTTPException(status_code=403, detail="Session does not belong to user")
        
        # Close the session
        success = await whisper_chatterbox_service.close_session(session_id)
        if success:
            return {"message": "Session closed successfully"}
        else:
            raise HTTPException(status_code=500, detail="Failed to close session")
            
    except Exception as e:
        logger.error(f"Error cleaning up session {session_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to cleanup session: {str(e)}")


@router.get("/health")
async def whisper_voice_health_check():
    """
    Health check endpoint for the Whisper + Chatterbox voice service.
    """
    try:
        # Check if Whisper model is loaded
        whisper_status = hasattr(whisper_chatterbox_service, 'whisper_model') and whisper_chatterbox_service.whisper_model is not None
        
        # Check TTS engine
        tts_status = hasattr(whisper_chatterbox_service, 'tts_engine') and whisper_chatterbox_service.tts_engine is not None
        
        # Check Redis connection
        redis_status = await whisper_chatterbox_service.redis_client.ping()
        
        # Check Gemini
        gemini_status = hasattr(whisper_chatterbox_service, 'gemini_model') and whisper_chatterbox_service.gemini_model is not None
        
        return {
            "status": "healthy" if all([whisper_status, tts_status, redis_status, gemini_status]) else "unhealthy",
            "service": "whisper_chatterbox",
            "components": {
                "whisper_asr": whisper_status,
                "tts_engine": tts_status,
                "tts_type": whisper_chatterbox_service.tts_type,
                "redis": redis_status,
                "gemini": gemini_status
            },
            "active_sessions": len(whisper_chatterbox_service.active_sessions)
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "whisper_chatterbox",
            "error": str(e)
        }


@router.get("/models")
async def get_available_models():
    """
    Get information about available models.
    """
    return {
        "asr": {
            "engine": "whisper",
            "model": whisper_chatterbox_service.whisper_model.dims.n_mels if hasattr(whisper_chatterbox_service, 'whisper_model') else "unknown",
            "device": whisper_chatterbox_service.whisper_device if hasattr(whisper_chatterbox_service, 'whisper_device') else "unknown"
        },
        "tts": {
            "engine": whisper_chatterbox_service.tts_type,
            "available": whisper_chatterbox_service.tts_engine is not None
        },
        "conversation": {
            "engine": "gemini",
            "model": "gemini-2.0-flash"
        }
    } 