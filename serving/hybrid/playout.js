// One gain envelope per utterance; native PCM packets stay unchanged.
export class AudioPlayout {
  constructor(context, receipt = () => {}) {
    this.context = context; this.receipt = receipt; this.events = new Map(); this.cancelled = new Set();
    this.sequence = 0; this.frames = 0; this.underruns = 0;
  }
  append(data) {
    if (data.byteLength !== 8 + 1920 * 4) throw new Error('Unexpected audio packet size');
    const header = new DataView(data), identity = header.getUint32(0, true), number = header.getUint32(4, true);
    if (number !== this.sequence++) throw new Error('Audio packet sequence changed');
    if (this.cancelled.has(identity)) return;
    const pcm = new Float32Array(data, 8);
    if (!pcm.every(Number.isFinite)) throw new Error('Nonfinite model audio');
    const context = this.context, buffer = context.createBuffer(1, pcm.length, 24000); buffer.copyToChannel(pcm, 0);
    let state = this.events.get(identity);
    if (!state) {
      const gain = context.createGain(); gain.connect(context.destination);
      state = { next: context.currentTime + .16, nodes: new Set(), gain };
      gain.gain.setValueAtTime(0, state.next); gain.gain.linearRampToValueAtTime(1, state.next + .005);
      this.events.set(identity, state);
    }
    if (state.next < context.currentTime) { this.underruns++; state.next = context.currentTime; }
    const node = context.createBufferSource(); node.buffer = buffer; node.connect(state.gain);
    const start = state.next; node.start(start); state.next += pcm.length / 24000; state.nodes.add(node);
    node.onended = () => state.nodes.delete(node); this.frames++;
    if (this.frames % 25 === 0) this.receipt({ kind: 'playback_receipt', output_frames: this.frames, last_sequence: number,
      event: identity, context_time: context.currentTime, scheduled_start: start, underruns: this.underruns, physical_device_recording: false });
  }
  interrupt(identity) {
    this.cancelled.add(identity); const state = this.events.get(identity);
    if (!state) return;
    const at = this.context.currentTime, gain = state.gain.gain;
    const value = gain.value; gain.cancelScheduledValues(at); gain.setValueAtTime(value, at); gain.linearRampToValueAtTime(0, at + .008);
    for (const node of state.nodes) try { node.stop(at + .008); } catch {}
    this.events.delete(identity);
    this.receipt({ kind: 'playback_receipt', event: identity, context_time: at, interruption_fade_ms: 8, physical_device_recording: false });
  }
  stop() { for (const identity of [...this.events.keys()]) this.interrupt(identity); }
}
