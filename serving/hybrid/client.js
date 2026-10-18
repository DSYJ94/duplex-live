'use strict';
const $ = (id) => document.getElementById(id);
$('token').value = new URLSearchParams(location.hash.slice(1)).get('token') || '';
let context, stream, socket, capture, source, muted, ready = false, sequence = 0, expectedOutput = 0;
let outputFrames = 0, underruns = 0;
const scheduled = new Map(), cancelled = new Set();
function status(text) { $('status').textContent = text; }
function append(who, text) { $('activity').textContent += `${who}: ${text}\n\n`; }
function stopPlayback() { for (const state of scheduled.values()) for (const node of state.nodes) try { node.stop(); } catch {} scheduled.clear(); }
function stop() {
  ready = false; capture?.disconnect(); source?.disconnect(); muted?.disconnect();
  stream?.getTracks().forEach((track) => track.stop()); stopPlayback();
  if (socket && socket.readyState < 2) socket.close(1000, 'Owner ended session');
  context?.close(); $('start').disabled = false; $('end').disabled = true; status('Session ended.');
}
function play(data) {
  if (data.byteLength !== 8 + 1920 * 4) throw new Error('Unexpected audio packet size');
  const header = new DataView(data), identity = header.getUint32(0, true), number = header.getUint32(4, true);
  if (number !== expectedOutput++) throw new Error('Audio packet sequence changed');
  if (cancelled.has(identity)) return;
  const pcm = new Float32Array(data, 8), buffer = context.createBuffer(1, pcm.length, 24000);
  buffer.copyToChannel(pcm, 0); const node = context.createBufferSource(); node.buffer = buffer; node.connect(context.destination);
  let state = scheduled.get(identity);
  if (!state) { state = { next: context.currentTime + .16, nodes: new Set() }; scheduled.set(identity, state); }
  if (state.next < context.currentTime) { underruns++; state.next = context.currentTime; }
  const start = state.next; node.start(start); state.next += pcm.length / 24000; state.nodes.add(node);
  node.onended = () => state.nodes.delete(node); outputFrames++;
  if (outputFrames % 25 === 0 && socket.readyState === WebSocket.OPEN) socket.send(JSON.stringify({ kind: 'playback_receipt', output_frames: outputFrames,
    last_sequence: number, event: identity, context_time: context.currentTime, scheduled_start: start, underruns, physical_device_recording: false }));
}
$('end').onclick = stop;
$('start').onclick = async () => {
  $('start').disabled = true; expectedOutput = 0; outputFrames = 0; underruns = 0; sequence = 0; cancelled.clear();
  try {
    if (!$('token').value) throw new Error('A session access code is required.');
    context = new AudioContext({ sampleRate: 24000 }); await context.resume();
    if (context.sampleRate !== 24000) throw new Error('This browser did not provide 24 kHz audio. Please use Chrome.');
    await context.audioWorklet.addModule('/worklet.js');
    stream = await navigator.mediaDevices.getUserMedia({ audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true } });
    const { deviceId, groupId, ...appliedSettings } = stream.getAudioTracks()[0].getSettings();
    const settings = { ...appliedSettings, audio_context_sample_rate: context.sampleRate, aec_requested: true, physical_aec_tested: false };
    $('settings').textContent = JSON.stringify(settings, null, 2);
    source = context.createMediaStreamSource(stream); capture = new AudioWorkletNode(context, 'capture-frames'); muted = context.createGain(); muted.gain.value = 0;
    capture.connect(muted).connect(context.destination);
    socket = new WebSocket(`${location.protocol === 'https:' ? 'wss:' : 'ws:'}//${location.host}/ws`); socket.binaryType = 'arraybuffer';
    socket.onopen = () => socket.send(JSON.stringify({ token: $('token').value, sample_rate: 24000, frame_samples: 1920, capture_settings: settings, client_kind: 'browser-microphone' }));
    capture.port.onmessage = ({ data }) => {
      if (!ready || socket.readyState !== WebSocket.OPEN) return;
      if (socket.bufferedAmount > 16 * 7684) { status('The audio connection is falling behind.'); stop(); return; }
      const packet = new ArrayBuffer(4 + data.byteLength); new DataView(packet).setUint32(0, sequence++, true); new Uint8Array(packet, 4).set(new Uint8Array(data)); socket.send(packet);
    };
    socket.onmessage = ({ data }) => {
      try {
        if (data instanceof ArrayBuffer) { play(data); return; }
        const event = JSON.parse(data);
        if (event.kind === 'session_ready') { ready = true; source.connect(capture); $('end').disabled = false; status('Listening — you can speak at any time.'); }
        if (event.kind === 'user_utterance_complete') append('You', event.text);
        if (event.kind === 'speech_started') append('Assistant', event.text);
        if (event.kind === 'speech_interrupted') {
          const identity = Number(event.event_id.replace('speech-', '')); cancelled.add(identity);
          const state = scheduled.get(identity); if (state) { for (const node of state.nodes) try { node.stop(); } catch {} scheduled.delete(identity); }
        }
        if (event.kind === 'session_error') { const message = event.message; stop(); status(`Session error: ${message}`); }
        if (event.kind === 'session_complete') status('Conversation complete.');
      } catch (error) { stop(); status(error.message); }
    };
    socket.onerror = () => { stop(); status('The voice connection failed.'); };
    socket.onclose = () => { if (ready) { stop(); status('The voice connection closed.'); } };
    status('Connecting to the voice service…');
  } catch (error) { stop(); status(error.message); }
};
