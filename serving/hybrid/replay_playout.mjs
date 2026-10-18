// Replay saved wire order through the production scheduling module in Node.
// This is a clock simulation, never browser-engine or physical-device evidence.
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import assert from 'node:assert/strict';
const folder = resolve(process.argv[2]);
const source = readFileSync(new URL('./playout.js', import.meta.url), 'utf8');
const { AudioPlayout } = await import(`data:text/javascript;base64,${Buffer.from(source).toString('base64')}`);
const events = JSON.parse(readFileSync(`${folder}/network-events.json`));
const arrivals = JSON.parse(readFileSync(`${folder}/packet-arrivals.json`));
const wire = readFileSync(`${folder}/received-wire-packets.bin`);
assert.equal(wire.length, arrivals.length * 7688);
const ordered = [...events.map(event => ({ ...event, type: 'event' })), ...arrivals.map((row, index) => ({ ...row, type: 'pcm', index }))].sort((a,b) => a.wire_arrival_index - b.wire_arrival_index);
ordered.forEach((row,i) => assert.equal(row.wire_arrival_index,i));
const origin = ordered[0].received_monotonic, nodes = [], buffers = [], receipts = [];
const context = { currentTime: 0, destination: {},
  createBuffer(_, length) { const b = { length, copyToChannel(pcm) { this.pcm = pcm.slice(); } }; buffers.push(b); return b; },
  createGain: () => ({ gain: { setValueAtTime() {}, linearRampToValueAtTime() {}, cancelAndHoldAtTime() {} }, connect() {}, disconnect() {} }),
  createBufferSource() { const n = { connect() {}, disconnect() {}, start(at, offset=0) { this.at=at;this.offset=offset; }, stop(at) { this.stoppedAt=at; } }; nodes.push(n); return n; }
};
const player = new AudioPlayout(context, row => receipts.push(row));
const identity = value => { assert.match(value,/^speech-[1-9][0-9]*$/);return Number(value.slice(7)); };
for (const row of ordered) {
  context.currentTime = row.received_monotonic - origin;
  assert.ok(Number.isFinite(context.currentTime) && context.currentTime >= 0);
  for (const node of nodes) if (!node.ended && node.at !== undefined && Math.min(node.at+(node.buffer.length/24000-node.offset),node.stoppedAt ?? Infinity)<=context.currentTime) { node.ended=true;node.onended(); }
  if (row.type === 'pcm') {
    const packet = wire.subarray(row.index*7688,(row.index+1)*7688);
    player.append(packet.buffer.slice(packet.byteOffset,packet.byteOffset+packet.byteLength));
  } else if (row.kind === 'speech_interrupted') {
    if (row.reason === 'acoustic_pause') player.pause(identity(row.event_id)); else player.interrupt(identity(row.event_id));
  } else if (row.kind === 'speech_pause_committed') player.interrupt(identity(row.event_id));
  else if (row.kind === 'speech_resumed') player.resume(identity(row.resumes_event_id),identity(row.event_id));
}
// Each source sample may have at most one scheduled playback window. Gain fades
// change amplitude, not sample ownership. Discarded suffixes remain visible.
let scheduledSamples = 0, discardedSamples = 0;
for (const buffer of buffers) {
  const windows = nodes.filter(n=>n.buffer===buffer).map(n=> {
    const first = Math.round(n.offset*24000);
    const played = Math.max(0,Math.min(buffer.length-first,Math.round(((n.stoppedAt ?? Infinity)-n.at)*24000)));
    return [first,first+played];
  }).filter(([a,b])=>b>a).sort((a,b)=>a[0]-b[0]);
  let end=0,total=0;
  for (const [first,last] of windows) { assert.ok(first>=end,'Duplicate source sample scheduled');end=last;total+=last-first; }
  assert.ok(total<=buffer.length);scheduledSamples+=total;discardedSamples+=buffer.length-total;
}
const report = { schema:'dl-client-playout-wire-replay-v1', production_module_executed:true, wire_order_exact:true,
  received_packets:arrivals.length, scheduled_source_samples:scheduledSamples, discarded_source_samples:discardedSamples,
  no_duplicate_source_samples:true, pause_count:receipts.filter(r=>r.action==='pause').length,
  resume_count:receipts.filter(r=>r.action==='resume').length, retained_samples_rescheduled:receipts.filter(r=>r.action==='resume').reduce((n,r)=>n+r.retained_source_samples,0),
  clock_kind:'simulated-audio-context-from-real-wire-arrivals', browser_engine_tested:false, physical_device_playback:false, duplex_qualified:false };
writeFileSync(`${folder}/client-playout-replay.json`,JSON.stringify({ ...report, receipts },null,2)+'\n');
console.log(JSON.stringify(report));
