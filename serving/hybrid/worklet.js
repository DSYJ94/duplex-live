class CaptureFrames extends AudioWorkletProcessor {
  constructor() { super(); this.frame = new Float32Array(1920); this.offset = 0; }
  process(inputs, outputs) {
    for (const channels of outputs) for (const channel of channels) channel.fill(0);
    const mono = inputs[0]?.[0];
    if (mono) for (let i = 0; i < mono.length; i++) {
      this.frame[this.offset++] = mono[i];
      if (this.offset === this.frame.length) {
        this.port.postMessage(this.frame.buffer, [this.frame.buffer]);
        this.frame = new Float32Array(1920); this.offset = 0;
      }
    }
    return true;
  }
}
registerProcessor('capture-frames', CaptureFrames);
