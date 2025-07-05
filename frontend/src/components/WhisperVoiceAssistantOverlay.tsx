'use client';

import { useState, useRef, useCallback, useEffect } from 'react';
import { FaPause, FaPlay, FaTimes, FaMicrophone, FaVolumeUp } from 'react-icons/fa';
import { useMicrophoneVolume } from '@/hooks/useMicrophoneVolume';
import {
  startAudioPlayerWorklet,
  startAudioRecorderWorklet,
  base64ToArray,
} from '@/services/audioProcessor';
import { getAccessToken } from '@/services/api';
import { useAuth } from '@/context/AuthProvider';

interface WhisperVoiceEvent {
  type: 'text' | 'audio' | 'transcription' | 'session_created' | 'service_info' | 'heartbeat' | 'error';
  data?: string;
  mime_type?: string;
  text?: string;
  session_id?: string;
  timestamp?: string;
  error?: string;
  message?: string;
  partial?: boolean;
  confidence?: number;
  service?: string;
  models?: { asr: string; tts: string };
}

interface WhisperVoiceAssistantOverlayProps {
  isOpen: boolean;
  onClose: () => void;
  isDarkMode: boolean;
}

export default function WhisperVoiceAssistantOverlay({ isOpen, onClose, isDarkMode }: WhisperVoiceAssistantOverlayProps) {
  const [paused, setPaused] = useState(false);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isListening, setIsListening] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [lastResponse, setLastResponse] = useState<string>('');
  const [transcribedText, setTranscribedText] = useState<string>('');
  const [sessionId, setSessionId] = useState<string>('');
  const [serviceInfo, setServiceInfo] = useState<{ asr: string; tts: string } | null>(null);
  const [confidence, setConfidence] = useState<number>(0);
  const [error, setError] = useState<string>('');
  
  const { user } = useAuth();
  const userId = user?.id || '';
  
  // Audio system refs
  const audioPlayerNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioPlayerContextRef = useRef<AudioContext | null>(null);
  const audioRecorderNodeRef = useRef<AudioWorkletNode | null>(null);
  const audioRecorderContextRef = useRef<AudioContext | null>(null);
  
  // Audio buffer for batching
  const audioBufferRef = useRef<Uint8Array[]>([]);
  const isSendingRef = useRef(false);
  const sessionIdRef = useRef<string>('');
  const speakingTimeoutRef = useRef<NodeJS.Timeout | null>(null);
  
  // Microphone volume hook
  const { volume, startListening, stopListening } = useMicrophoneVolume();
  
  // Update session ID ref when it changes
  useEffect(() => {
    sessionIdRef.current = sessionId;
  }, [sessionId]);
  
  // Detect user interruption based on volume
  useEffect(() => {
    if (volume > 0.05 && isSpeaking && !paused && isOpen) {
      console.log('🎤 User interrupted with voice - switching to listening');
      setIsSpeaking(false);
      
      // Clear speaking timeout
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current);
        speakingTimeoutRef.current = null;
      }
    }
  }, [volume, isSpeaking, paused, isOpen]);
  
  // Initialize SSE connection
  useEffect(() => {
    if (isOpen && userId) {
      initializeWhisperVoiceMode();
    }
    return () => {
      if (!isOpen) {
        cleanup();
      }
    };
  }, [isOpen, userId]);
  
  const initializeWhisperVoiceMode = useCallback(async () => {
    try {
      console.log('🎤 Whisper voice mode starting...');
      
      // Initialize audio system
      const audioInitialized = await initializeAudio();
      if (!audioInitialized) {
        setError('Failed to initialize audio system');
        return;
      }
      
      // Start microphone monitoring
      startListening();
      
      // Start SSE connection
      const token = await getAccessToken();
      const audioMode = true;
      const sseUrl = `${process.env.NEXT_PUBLIC_API_URL}/api/voice-whisper/events/${userId}?is_audio=${audioMode}`;
      
      const eventSource = new EventSource(sseUrl, {
        withCredentials: false,
      });
      
      eventSource.addEventListener('message', (event) => {
        try {
          const messageFromServer: WhisperVoiceEvent = JSON.parse(event.data);
          handleServerMessage(messageFromServer);
        } catch (error) {
          console.error('Error parsing SSE message:', error);
        }
      });
      
      eventSource.addEventListener('error', (error) => {
        console.error('SSE connection error:', error);
        setError('Connection error occurred');
      });
      
      // Store event source for cleanup
      (window as any).whisperVoiceEventSource = eventSource;
      
    } catch (error) {
      console.error('❌ Failed to start Whisper voice mode:', error);
      setError('Failed to start voice mode');
    }
  }, [userId, startListening]);
  
  const handleServerMessage = (message: WhisperVoiceEvent) => {
    console.log('📨 Received message:', message);
    
    if (message.type === 'session_created') {
      setSessionId(message.session_id || '');
      console.log('✅ Whisper voice session created:', message.session_id);
      
    } else if (message.type === 'service_info') {
      setServiceInfo(message.models || null);
      console.log('ℹ️ Service info:', message.models);
      
    } else if (message.type === 'transcription') {
      setTranscribedText(message.text || '');
      setConfidence(message.confidence || 0);
      setIsTranscribing(false);
      setIsListening(false);
      console.log('📝 Transcription:', message.text, 'confidence:', message.confidence);
      
    } else if (message.type === 'text' && (message.data || message.text)) {
      const textContent = message.data || message.text || '';
      setLastResponse(textContent);
      setIsSpeaking(true);
      
      // Set timeout for speaking state
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current);
      }
      speakingTimeoutRef.current = setTimeout(() => {
        setIsSpeaking(false);
        setIsListening(true);
        console.log('🔇 Speaking timeout - back to listening');
      }, 8000);
      
    } else if (message.type === 'audio' && message.data) {
      // Play audio response
      if (audioPlayerNodeRef.current && message.mime_type) {
        setIsSpeaking(true);
        
        // Convert base64 to audio data based on mime type
        if (message.mime_type === 'audio/wav' || message.mime_type === 'audio/mp3') {
          // For WAV/MP3, we need to decode and play
          playAudioFromBase64(message.data);
        } else {
          // For PCM, use the existing method
          audioPlayerNodeRef.current.port.postMessage(base64ToArray(message.data));
        }
        
        // Set timeout for speaking state
        if (speakingTimeoutRef.current) {
          clearTimeout(speakingTimeoutRef.current);
        }
        speakingTimeoutRef.current = setTimeout(() => {
          setIsSpeaking(false);
          setIsListening(true);
          console.log('🔇 Audio playback timeout - back to listening');
        }, 10000);
      }
      
    } else if (message.type === 'error') {
      setError(message.message || 'Unknown error occurred');
      console.error('❌ Server error:', message.message);
    }
  };
  
  const playAudioFromBase64 = async (base64Data: string) => {
    try {
      // Decode base64 to binary
      const binaryString = atob(base64Data);
      const bytes = new Uint8Array(binaryString.length);
      for (let i = 0; i < binaryString.length; i++) {
        bytes[i] = binaryString.charCodeAt(i);
      }
      
      // Create audio context if not exists
      if (!audioPlayerContextRef.current) {
        audioPlayerContextRef.current = new AudioContext();
      }
      
      // Decode audio data
      const audioBuffer = await audioPlayerContextRef.current.decodeAudioData(bytes.buffer);
      
      // Create and play audio source
      const source = audioPlayerContextRef.current.createBufferSource();
      source.buffer = audioBuffer;
      source.connect(audioPlayerContextRef.current.destination);
      
      // Handle playback end
      source.onended = () => {
        setIsSpeaking(false);
        setIsListening(true);
        if (speakingTimeoutRef.current) {
          clearTimeout(speakingTimeoutRef.current);
          speakingTimeoutRef.current = null;
        }
        console.log('🔇 Audio playback ended');
      };
      
      source.start();
      
    } catch (error) {
      console.error('❌ Error playing audio:', error);
      setError('Failed to play audio response');
    }
  };
  
  // Initialize audio system
  const initializeAudio = useCallback(async () => {
    try {
      // Start audio output
      const [playerNode, playerCtx] = await startAudioPlayerWorklet();
      audioPlayerNodeRef.current = playerNode as AudioWorkletNode;
      audioPlayerContextRef.current = playerCtx as AudioContext;
      
      // Listen for audio playback completion
      if (audioPlayerNodeRef.current) {
        audioPlayerNodeRef.current.port.onmessage = (event) => {
          if (event.data.type === 'playback_ended') {
            setIsSpeaking(false);
            setIsListening(true);
            if (speakingTimeoutRef.current) {
              clearTimeout(speakingTimeoutRef.current);
              speakingTimeoutRef.current = null;
            }
            console.log('🔇 Audio playback ended');
          }
        };
      }
      
      // Start audio input
      const [recorderNode, recorderCtx] = await startAudioRecorderWorklet(audioRecorderHandler);
      audioRecorderNodeRef.current = recorderNode as AudioWorkletNode;
      audioRecorderContextRef.current = recorderCtx as AudioContext;
      
      console.log('🎵 Audio system initialized successfully');
      setIsListening(true);
      return true;
    } catch (error) {
      console.error('❌ Failed to initialize audio system:', error);
      return false;
    }
  }, []);
  
  // Handle audio data from recorder
  function audioRecorderHandler(pcmData: ArrayBuffer) {
    if (paused || !isOpen || isSpeaking) return;
    
    audioBufferRef.current.push(new Uint8Array(pcmData));
    
    // If the sending loop is not running, start it
    if (!isSendingRef.current) {
      sendBufferedAudio();
    }
  }
  
  // Send buffered audio to server
  const sendBufferedAudio = useCallback(async () => {
    if (paused || !isOpen || isSpeaking) {
      isSendingRef.current = false;
      return;
    }
    
    const currentSessionId = sessionIdRef.current || sessionId;
    if (!currentSessionId) {
      console.warn('⚠️ No sessionId available, cannot send audio');
      isSendingRef.current = false;
      return;
    }
    
    if (audioBufferRef.current.length === 0) {
      isSendingRef.current = false;
      return;
    }
    
    isSendingRef.current = true;
    setIsTranscribing(true);
    
    // Combine all audio chunks into one
    const audioBufferToSend = [...audioBufferRef.current];
    audioBufferRef.current = []; // Clear the buffer
    
    const totalLength = audioBufferToSend.reduce((acc, chunk) => acc + chunk.length, 0);
    const combinedArray = new Uint8Array(totalLength);
    let offset = 0;
    for (const chunk of audioBufferToSend) {
      combinedArray.set(chunk, offset);
      offset += chunk.length;
    }
    
    const base64Data = btoa(String.fromCharCode(...combinedArray));
    
    console.log(`🎤 Sending audio to Whisper ASR: ${base64Data.length} chars`);
    
    try {
      const token = await getAccessToken();
      const response = await fetch(`${process.env.NEXT_PUBLIC_API_URL}/api/voice-whisper/send/${currentSessionId}`, {
        method: 'POST',
        headers: { 
          'Content-Type': 'application/json',
          'Authorization': `Bearer ${token}`
        },
        body: JSON.stringify({ mime_type: 'audio/pcm', data: base64Data }),
      });
      
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}: ${response.statusText}`);
      }
      
      // Handle streaming response
      const reader = response.body?.getReader();
      if (reader) {
        try {
          while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            
            const chunk = new TextDecoder().decode(value);
            const lines = chunk.split('\n');
            
            for (const line of lines) {
              if (line.startsWith('data: ')) {
                try {
                  const data = JSON.parse(line.substring(6));
                  handleServerMessage(data);
                } catch {
                  console.warn('Failed to parse chunk:', line);
                }
              }
            }
          }
        } finally {
          reader.releaseLock();
        }
      }
    } catch (error) {
      console.error('Failed to send audio data:', error);
      setError('Failed to send audio');
      // Put the unsent audio back at the beginning of the buffer
      audioBufferRef.current = [...audioBufferToSend, ...audioBufferRef.current];
    } finally {
      isSendingRef.current = false;
      setIsTranscribing(false);
      // Continue sending if there's more data
      if (audioBufferRef.current.length > 0) {
        setTimeout(sendBufferedAudio, 500);
      }
    }
  }, [isOpen, paused, sessionId, isSpeaking]);
  
  const handlePauseResume = () => {
    setPaused(!paused);
    if (!paused) {
      console.log('⏸️ Whisper voice paused');
      stopListening();
    } else {
      console.log('▶️ Whisper voice resumed');
      startListening();
    }
  };
  
  const handleClose = () => {
    cleanup();
    onClose();
  };
  
  const cleanup = () => {
    try {
      // Stop microphone monitoring
      stopListening();
      
      // Close SSE connection
      if ((window as any).whisperVoiceEventSource) {
        (window as any).whisperVoiceEventSource.close();
        (window as any).whisperVoiceEventSource = null;
      }
      
      // Stop audio contexts
      if (audioPlayerContextRef.current) {
        audioPlayerContextRef.current.close();
        audioPlayerContextRef.current = null;
      }
      if (audioRecorderContextRef.current) {
        audioRecorderContextRef.current.close();
        audioRecorderContextRef.current = null;
      }
      
      // Clear timeouts
      if (speakingTimeoutRef.current) {
        clearTimeout(speakingTimeoutRef.current);
        speakingTimeoutRef.current = null;
      }
      
      // Reset states
      setPaused(false);
      setIsSpeaking(false);
      setIsListening(false);
      setIsTranscribing(false);
      setLastResponse('');
      setTranscribedText('');
      setSessionId('');
      setServiceInfo(null);
      setConfidence(0);
      setError('');
      
      console.log('🔚 Whisper voice assistant closed');
    } catch (error) {
      console.error('Error during cleanup:', error);
    }
  };
  
  if (!isOpen) return null;
  
  // Determine current status
  let statusText = 'Initializing...';
  let statusIcon = <FaMicrophone className="text-blue-400" />;
  
  if (error) {
    statusText = error;
    statusIcon = <FaTimes className="text-red-400" />;
  } else if (isTranscribing) {
    statusText = 'Transcribing...';
    statusIcon = <FaMicrophone className="text-yellow-400 animate-pulse" />;
  } else if (isSpeaking) {
    statusText = 'Speaking...';
    statusIcon = <FaVolumeUp className="text-green-400 animate-pulse" />;
  } else if (isListening) {
    statusText = 'Listening...';
    statusIcon = <FaMicrophone className="text-blue-400" />;
  } else if (paused) {
    statusText = 'Paused';
    statusIcon = <FaPause className="text-gray-400" />;
  }
  
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black bg-opacity-50">
      <div className={`relative w-96 h-96 rounded-full ${isDarkMode ? 'bg-gray-800' : 'bg-white'} shadow-2xl flex flex-col items-center justify-center`}>
        {/* Close button */}
        <button
          onClick={handleClose}
          className="absolute top-4 right-4 p-2 rounded-full hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors"
        >
          <FaTimes className="text-gray-600 dark:text-gray-300" />
        </button>
        
        {/* Service info */}
        {serviceInfo && (
          <div className="absolute top-4 left-4 text-xs text-gray-500 dark:text-gray-400">
            <div>ASR: {serviceInfo.asr}</div>
            <div>TTS: {serviceInfo.tts}</div>
          </div>
        )}
        
        {/* Central animation area */}
        <div className="flex-1 flex items-center justify-center">
          <div className="relative">
            {/* Volume visualization */}
            <div
              className={`w-32 h-32 rounded-full border-4 transition-all duration-300 ${
                isListening && !paused
                  ? 'border-blue-400 animate-pulse'
                  : isSpeaking
                  ? 'border-green-400 animate-pulse'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
              style={{
                transform: `scale(${1 + (isListening ? volume * 0.5 : 0)})`,
              }}
            >
              <div className="w-full h-full rounded-full bg-gradient-to-r from-blue-400 to-purple-500 opacity-20"></div>
            </div>
            
            {/* Status icon */}
            <div className="absolute inset-0 flex items-center justify-center">
              <div className="text-4xl">{statusIcon}</div>
            </div>
          </div>
        </div>
        
        {/* Status text */}
        <div className="mb-4 text-center">
          <div className="text-lg font-semibold text-gray-800 dark:text-gray-200">
            {statusText}
          </div>
          {transcribedText && (
            <div className="text-sm text-gray-600 dark:text-gray-400 mt-1 max-w-xs truncate">
              "{transcribedText}" {confidence > 0 && `(${(confidence * 100).toFixed(0)}%)`}
            </div>
          )}
          {lastResponse && (
            <div className="text-xs text-gray-500 dark:text-gray-500 mt-1 max-w-xs truncate">
              {lastResponse}
            </div>
          )}
        </div>
        
        {/* Control buttons */}
        <div className="mb-8 flex space-x-4">
          <button
            onClick={handlePauseResume}
            disabled={error !== ''}
            className={`p-3 rounded-full transition-colors ${
              paused
                ? 'bg-green-500 hover:bg-green-600 text-white'
                : 'bg-yellow-500 hover:bg-yellow-600 text-white'
            } ${error ? 'opacity-50 cursor-not-allowed' : ''}`}
          >
            {paused ? <FaPlay /> : <FaPause />}
          </button>
        </div>
      </div>
    </div>
  );
} 