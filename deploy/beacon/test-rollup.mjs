/**
 * Self-check for scheduled()'s hourly compaction.  node test-rollup.mjs [dir]
 *
 * The one thing that must never drift: rollupHour() and the QUALIFY in
 * headroom-beacon-stats/beacon.sh have to agree on which heartbeat wins. If
 * they disagree the reports get quietly wrong rather than loudly broken, so
 * this asserts the JS picks exactly the max-seq row per (install, session).
 *
 * Point it at a directory of real beacon objects to check against the corpus:
 *   aws s3 sync s3://headroom-telemetry/sessions/dt=.../hh=.../ /tmp/hr/ ...
 *   node test-rollup.mjs /tmp/hr
 * With no argument it runs on a small fixture and needs no network.
 */
import { readdirSync, readFileSync } from 'node:fs';
import assert from 'node:assert/strict';
import { rollupHour } from './worker.js';

/** The slice of the R2 binding rollupHour uses, backed by a plain object. */
function stubBucket(files) {
  const written = {};
  return {
    written,
    list: async ({ prefix }) => ({
      objects: Object.keys(files)
        .filter((k) => k.startsWith(prefix))
        .map((key) => ({ key })),
      truncated: false,
    }),
    get: async (key) => ({ text: async () => files[key] }),
    put: async (key, body) => {
      written[key] = body;
    },
  };
}

const beacon = (install, id, seq) =>
  JSON.stringify({ resource: { 'headroom.install_id': install }, session: { id, seq } });

async function run(files, part) {
  const CORPUS = stubBucket(files);
  const read = await rollupHour({ CORPUS }, part);
  const out = CORPUS.written[`rollup/${part}/data.ndjson`];
  return { read, rows: out ? out.split('\n').map((l) => JSON.parse(l)) : [] };
}

const PART = 'dt=2026-08-06/hh=14';

// 1. Highest seq wins, out-of-order input, one row per (install, session).
{
  const files = {
    [`sessions/${PART}/a.json`]: [beacon('i1', 's1', 3), beacon('i1', 's2', 1)].join('\n'),
    [`sessions/${PART}/b.json`]: beacon('i1', 's1', 9),
    [`sessions/${PART}/c.json`]: beacon('i1', 's1', 7),
    // Same session id under a different install must not collapse together.
    [`sessions/${PART}/d.json`]: beacon('i2', 's1', 2),
  };
  const { read, rows } = await run(files, PART);
  assert.equal(read, 4, 'reads every object in the hour');
  assert.equal(rows.length, 3, 'one row per (install, session)');
  const seq = Object.fromEntries(
    rows.map((r) => [`${r.resource['headroom.install_id']} ${r.session.id}`, r.session.seq])
  );
  assert.deepEqual(seq, { 'i1 s1': 9, 'i1 s2': 1, 'i2 s1': 2 });
}

// 2. An unparseable object loses itself, never the rest of the hour, and an
//    empty hour writes nothing rather than a zero-byte object.
{
  const files = {
    [`sessions/${PART}/a.json`]: '{ this is not json',
    [`sessions/${PART}/b.json`]: `\n${beacon('i1', 's1', 4)}\n`,
  };
  const { rows } = await run(files, PART);
  assert.deepEqual(rows.map((r) => r.session.seq), [4], 'survives a corrupt object');

  const empty = await run({}, PART);
  assert.deepEqual(empty.rows, [], 'empty hour writes no object');
}

// 3. Against real objects, if a directory was given: same answer as the QUALIFY
//    in beacon.sh, which is `count(DISTINCT install||session)` rows, each
//    carrying that pair's max seq.
const dir = process.argv[2];
if (dir) {
  const files = {};
  for (const f of readdirSync(dir).filter((f) => f.endsWith('.json'))) {
    files[`sessions/${PART}/${f}`] = readFileSync(`${dir}/${f}`, 'utf8');
  }
  const { read, rows } = await run(files, PART);

  const expected = new Map();
  for (const text of Object.values(files)) {
    for (const line of text.split('\n')) {
      if (!line.trim()) continue;
      const r = JSON.parse(line);
      const k = `${r.resource?.['headroom.install_id']} ${r.session?.id}`;
      expected.set(k, Math.max(expected.get(k) ?? -1, r.session?.seq ?? 0));
    }
  }
  assert.equal(read, Object.keys(files).length);
  assert.equal(rows.length, expected.size, 'row count matches DISTINCT sessions');
  for (const r of rows) {
    const k = `${r.resource['headroom.install_id']} ${r.session.id}`;
    assert.equal(r.session.seq, expected.get(k), `max seq for ${k}`);
  }
  console.log(`real corpus: ${read} objects -> ${rows.length} sessions in 1 object`);
}

console.log('ok');
