// Pure scheduling tests of the production module; not a browser/device test.
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { test } from 'node:test';
const moduleSource = readFileSync(new URL('../../serving/hybrid/playout.js', import.meta.url), 'utf8');
const { AudioPlayout } = await import(`data:text/javascript;base64,${Buffer.from(moduleSource).toString('base64')}`);
function setup() {
  const nodes = [], receipts = [];
  const context = { currentTime: 0, destination: {},
    createBuffer: (_, length) => ({ length, copyToChannel(pcm) { this.pcm = pcm.slice(); } }),
    createGain: () => ({ gain: { setValueAtTime() {}, linearRampToValueAtTime() {}, cancelAndHoldAtTime() {} }, connect() {}, disconnect() {} }),
    createBufferSource() { const node = { connect() {}, disconnect() {}, start(at, offset = 0) { this.at = at; this.offset = offset; }, stop(at) { this.stoppedAt = at; } }; nodes.push(node); return node; }
  };
  const player = new AudioPlayout(context, r => receipts.push(r));
  return { context, nodes, receipts, player };
}
function packet(identity, sequence, value = sequence + 1) {
  const data = new ArrayBuffer(7688), header = new DataView(data);
  header.setUint32(0, identity, true); header.setUint32(4, sequence, true);
  new Float32Array(data, 8).fill(value); return data;
}
test('pause retains exactly the client-scheduled suffix and queued packets once', () => {
  const { context, player, nodes } = setup();
  player.append(packet(1, 0)); player.append(packet(1, 1));
  context.currentTime = .2; player.pause(1); player.pause(1);
  assert.equal(nodes[0].stoppedAt, .208);
  player.append(packet(1, 2)); context.currentTime = .4; player.resume(1, 2);
  assert.equal(nodes.length, 5);
  assert.equal(nodes[2].buffer, nodes[0].buffer);
  assert.equal(Math.round(nodes[2].offset * 24000), 1152);
  assert.equal(Math.round(nodes[2].at * 24000), 13440);
  assert.equal(Math.round(nodes[3].at * 24000), 14208);
  assert.equal(nodes[3].offset, 0); assert.equal(nodes[4].offset, 0);
  nodes[0].onended(); // Late callback must not remove resumed nodes.
  player.append(packet(1, 3)); assert.equal(nodes.length, 5);
  player.append(packet(2, 4)); assert.equal(nodes.length, 6);
  player.interrupt(2); assert.equal(nodes[2].stoppedAt, .408);
});
test('hard cancellation cannot be resurrected by a late resume', () => {
  const { player, nodes } = setup(); player.append(packet(1, 0)); player.pause(1); player.interrupt(1);
  player.resume(1, 2); player.append(packet(2, 1)); assert.equal(nodes.length, 1);
});
test('pause before first PCM and session stop preserve cancellation', () => {
  const { player, nodes } = setup(); player.pause(1); player.append(packet(1, 0)); assert.equal(nodes.length, 0);
  player.resume(1, 2); assert.equal(nodes.length, 1); player.stop(); player.append(packet(2, 1)); assert.equal(nodes.length, 1);
});
test('invalid protocol and nonfinite cancelled packets fail closed', () => {
  const { player } = setup(); assert.throws(() => player.resume(1, 2));
  assert.throws(() => player.append(packet(0, 0)));
  const other = setup().player; other.interrupt(1);
  assert.throws(() => other.append(packet(1, 0, NaN)));
  const sequence = setup().player; assert.throws(() => sequence.append(packet(1, 1)));
});
test('retained audio is bounded and duplicate destinations cannot overwrite speech', () => {
  const { player } = setup(); player.pause(1);
  for (let i = 0; i < 75; i++) player.append(packet(1, i));
  assert.throws(() => player.append(packet(1, 75)), /limit/);
  const other = setup().player; other.pause(1); other.append(packet(2, 0));
  assert.throws(() => other.resume(1, 2));
});
