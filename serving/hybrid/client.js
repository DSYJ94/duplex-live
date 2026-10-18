import { AudioPlayout } from '/playout.js';
'use strict';
const $ = (id) => document.getElementById(id);
$('token').value = new URLSearchParams(location.hash.slice(1)).get('token') || '';
let context, stream, socket, capture, source, muted, playout, ready = false, sequence = 0, generation = 0;
let intentionalStop = false;
function status(text) { $('status').textContent = text; }
function append(who, text) { $('activity').textContent += `${who}: ${text}\n\n`; }
function stopPlayback() { playout?.stop(); }
function stop() {
  generation++; intentionalStop = true; ready = false; capture?.disconnect(); source?.disconnect(); muted?.disconnect();
  stream?.getTracks().forEach((track) => track.stop()); stopPlayback();
  if (socket && socket.readyState < 2) socket.close(1000, 'Owner ended session');
  const closing = context; context = undefined; setTimeout(() => closing?.close(), 12); $('start').disabled = false; $('end').disabled = true; status('Session ended.');
}
function play(data) { playout.append(data); }
$('end').onclick = stop;
$('start').onclick = async () => {
  const currentGeneration = ++generation; intentionalStop = false; $('start').disabled = true; sequence = 0;
  try {
    let endpoint = `${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`;
    const access = await fetch('/session-access', { cache: 'no-store' });
    if (access.ok) {
      const config = await access.json(); $('token').value = config.token; endpoint = config.websocket_url;
    } else if (!$('token').value) throw new Error('Session access is unavailable.');
    context = new AudioContext({ sampleRate: 24000 }); await context.resume();
    if (context.sampleRate !== 24000) throw new Error('This browser did not provide 24 kHz audio. Please use Chrome.');
    await context.audioWorklet.addModule('/worklet.js');
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const { deviceId, groupId, ...appliedSettings } = stream.getAudioTracks()[0].getSettings();
    const settings = { ...appliedSettings, audio_context_sample_rate: context.sampleRate, aec_requested: true, physical_aec_tested: false };
    $('settings').textContent = JSON.stringify(settings, null, 2);
    source = context.createMediaStreamSource(stream); capture = new AudioWorkletNode(context, 'capture-frames'); muted = context.createGain(); muted.gain.value = 0;
    capture.connect(muted).connect(context.destination);
    playout = new AudioPlayout(context, (receipt) => { if (currentGeneration === generation && socket?.readyState === WebSocket.OPEN) socket.send(JSON.stringify(receipt)); });
    socket = new WebSocket(endpoint); socket.binaryType = 'arraybuffer';
    socket.onopen = () => { if (currentGeneration !== generation) return; socket.send(JSON.stringify({ token: $('token').value, sample_rate: 24000, frame_samples: 1920, capture_settings: settings, client_kind: 'browser-microphone', playback_protocol: 'client-buffer-v1' })); };
    capture.port.onmessage = ({ data }) => {
      if (currentGeneration !== generation || !ready || socket.readyState !== WebSocket.OPEN) return;
      if (socket.bufferedAmount > 16 * 7684) { status('The audio connection is falling behind.'); stop(); return; }
      const packet = new ArrayBuffer(4 + data.byteLength); new DataView(packet).setUint32(0, sequence++, true); new Uint8Array(packet, 4).set(new Uint8Array(data)); socket.send(packet);
    };
    socket.onmessage = ({ data }) => {
      if (currentGeneration !== generation) return;
      try {
        if (data instanceof ArrayBuffer) { play(data); return; }
        const event = JSON.parse(data);
        if (event.kind === 'session_ready') { ready = true; source.connect(capture); $('end').disabled = false; status('Listening — you can speak at any time.'); }
        if (event.kind === 'user_utterance_complete') append('You', event.text);
        if (event.kind === 'speech_started' && !event.same_generator) append('Assistant', event.text);
        if (event.kind === 'speech_interrupted') {
          const identity = Number(event.event_id.replace('speech-', ''));
          if (event.reason === 'acoustic_pause') playout.pause(identity); else playout.interrupt(identity);
        }
        if (event.kind === 'speech_pause_committed') playout.interrupt(Number(event.event_id.replace('speech-', '')));
        if (event.kind === 'speech_resumed') playout.resume(Number(event.resumes_event_id.replace('speech-', '')), Number(event.event_id.replace('speech-', '')));
        if (event.kind === 'session_error') { const message = event.message; stop(); status(`Session error: ${message}`); }
        if (event.kind === 'session_complete') status('Conversation complete.');
      } catch (error) { stop(); status(error.message); }
    };
    socket.onerror = () => { if (currentGeneration !== generation) return; stop(); status('The voice connection failed.'); };
    socket.onclose = (event) => { if (currentGeneration === generation && !intentionalStop) { stop(); status(event.reason || 'The voice connection closed.'); } };
    status('Connecting to the voice service…');
  } catch (error) { if (currentGeneration === generation) { stop(); status(error.message); } }
};
