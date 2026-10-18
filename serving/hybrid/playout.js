// Client-owned scheduling positions; receipts never assert physical hearing.
const RATE = 24000, LEAD = 3840, FADE = 192, MAX_HELD = RATE * 6;
export class AudioPlayout {
  constructor(context, receipt = () => {}) {
    this.context = context; this.receipt = receipt; this.events = new Map(); this.cancelled = new Set();
    this.sequence = 0; this.frames = 0; this.underruns = 0;
  }
  _identity(identity) { if (!Number.isInteger(identity) || identity <= 0 || identity > 0xffffffff) throw new Error('Invalid speech identity'); }
  _state() { return { paused: false, held: [], heldSamples: 0, segment: null }; }
  _schedule(state, buffer, firstSample = 0) {
    const context = this.context, current = Math.round(context.currentTime * RATE);
    if (!state.segment) {
      const gain = context.createGain(); gain.connect(context.destination);
      state.segment = { next: current + LEAD, nodes: new Map(), gain, retired: false };
      gain.gain.setValueAtTime(0, state.segment.next / RATE);
      gain.gain.linearRampToValueAtTime(1, (state.segment.next + 120) / RATE);
    }
    const segment = state.segment;
    if (segment.next < current) { this.underruns++; segment.next = current; }
    const node = context.createBufferSource(); node.buffer = buffer; node.connect(segment.gain);
    const start = segment.next; segment.nodes.set(node, { buffer, start, firstSample });
    node.onended = () => {
      segment.nodes.delete(node); node.disconnect();
      if (segment.retired && !segment.nodes.size) segment.gain.disconnect();
    };
    node.start(start / RATE, firstSample / RATE); segment.next += buffer.length - firstSample;
    return start / RATE;
  }
  _hold(state, buffer, firstSample) {
    const count = buffer.length - firstSample;
    if (state.heldSamples + count > MAX_HELD) throw new Error('Paused audio retention limit reached');
    state.held.push({ buffer, firstSample }); state.heldSamples += count;
  }
  _retire(segment) {
    if (!segment) return;
    const at = this.context.currentTime, cut = Math.round(at * RATE) + FADE;
    segment.retired = true; segment.gain.gain.cancelAndHoldAtTime(at);
    segment.gain.gain.linearRampToValueAtTime(0, cut / RATE);
    for (const node of segment.nodes.keys()) node.stop(cut / RATE);
    if (!segment.nodes.size) segment.gain.disconnect();
  }
  append(data) {
    if (data.byteLength !== 8 + 1920 * 4) throw new Error('Unexpected audio packet size');
    const header = new DataView(data), identity = header.getUint32(0, true), number = header.getUint32(4, true);
    this._identity(identity);
    if (number !== this.sequence++) throw new Error('Audio packet sequence changed');
    const pcm = new Float32Array(data, 8);
    if (!pcm.every(Number.isFinite)) throw new Error('Nonfinite model audio');
    if (this.cancelled.has(identity)) return;
    const buffer = this.context.createBuffer(1, pcm.length, RATE); buffer.copyToChannel(pcm, 0);
    let state = this.events.get(identity);
    if (!state) { state = this._state(); this.events.set(identity, state); }
    const start = state.paused ? (this._hold(state, buffer, 0), null) : this._schedule(state, buffer);
    this.frames++;
    if (this.frames % 25 === 0) this.receipt({ kind: 'playback_receipt', output_frames: this.frames, last_sequence: number,
      event: identity, context_time: this.context.currentTime, scheduled_start: start, underruns: this.underruns, physical_device_recording: false });
  }
  pause(identity) {
    this._identity(identity); if (this.cancelled.has(identity)) return;
    let state = this.events.get(identity);
    if (!state) { state = this._state(); this.events.set(identity, state); }
    if (state.paused) return;
    const cut = Math.round(this.context.currentTime * RATE) + FADE;
    if (state.segment) for (const { buffer, start, firstSample } of state.segment.nodes.values()) {
      const first = firstSample + Math.max(0, Math.min(buffer.length - firstSample, cut - start));
      if (first < buffer.length) this._hold(state, buffer, first);
    }
    this._retire(state.segment); state.segment = null; state.paused = true;
    this.receipt({ kind: 'playback_receipt', event: identity, action: 'pause', retained_source_samples: state.heldSamples,
      context_time: this.context.currentTime, source_cursor_authority: 'client-audio-context', physical_device_recording: false });
  }
  resume(identity, destination) {
    this._identity(identity); this._identity(destination);
    if (identity === destination || this.events.has(destination) || this.cancelled.has(destination)) throw new Error('Invalid resumed speech identity');
    if (this.cancelled.has(identity)) { this.cancelled.add(destination); return; }
    const state = this.events.get(identity);
    if (!state?.paused) throw new Error('Speech resumed without a client pause');
    this.events.delete(identity); this.cancelled.add(identity); this.events.set(destination, state); state.paused = false;
    const retained = state.heldSamples;
    for (const { buffer, firstSample } of state.held) this._schedule(state, buffer, firstSample);
    state.held = []; state.heldSamples = 0;
    this.receipt({ kind: 'playback_receipt', event: destination, resumes_event: identity, action: 'resume', retained_source_samples: retained,
      context_time: this.context.currentTime, source_cursor_authority: 'client-audio-context', physical_device_recording: false });
  }
  interrupt(identity) {
    this._identity(identity); this.cancelled.add(identity); const state = this.events.get(identity);
    if (!state) return;
    this._retire(state.segment); this.events.delete(identity);
    this.receipt({ kind: 'playback_receipt', event: identity, context_time: this.context.currentTime, interruption_fade_ms: 8, physical_device_recording: false });
  }
  stop() { for (const identity of [...this.events.keys()]) this.interrupt(identity); }
}
