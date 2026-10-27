"""Provider-specific records for newly rendered streams; old receipts stay intact."""
from datetime import datetime,timezone

def record(engine_name,engine):
    at=datetime.now(timezone.utc).isoformat()
    if engine_name=='elevenlabs':
        return dict(tts_engine='ElevenLabs/'+engine.model,provenance_class='closed-content',
            license='closed-content training permitted under ADR0003 Amendment A; source retained',
            source_created_utc=at,provenance_authority='decisions/0003-provenance-classes.md#amendment-a-2026-09-30-closed-content-cleared-generally')
    return dict(tts_engine={'minicpm':'openbmb/MiniCPM-o-4_5','chatterbox':'ResembleAI/chatterbox'}[engine_name],
        provenance_class='open',license='synthetic open-model source; retain underlying source/voice attribution',source_created_utc=at)

def cache_namespace(engine_name,engine):
    # Legacy ElevenLabs cache names omitted model identity. Never assert a
    # current model ID for an old unpinned cached recording.
    return engine_name+'-model-'+engine.model if engine_name=='elevenlabs' else engine_name
