/**
 * Headroom telemetry beacon receiver.
 *
 * This file is open source on purpose. It is the other half of the promise
 * made in headroom/telemetry/session.py: users can read exactly what the
 * client sends AND exactly what happens to it on arrival. "Trust us" is not a
 * privacy policy.
 *
 * Deployed at otlp.headroomlabs.ai. Three jobs:
 *
 *   1. Allowlist. Drop every field not on ALLOWED_KEYS before anything is
 *      written. This is the only privacy control that works retroactively —
 *      if a future client version ships a bug that leaks a field, we cannot
 *      patch the installs already in the wild, but we can stop storing it
 *      here in one deploy.
 *
 *   2. Flatten. OTLP AnyValue nesting is portable but miserable to query
 *      ({"kvlistValue":{"values":[{"key":"tokens",...}]}}). We keep OTLP on
 *      the wire so the backend stays vendor-swappable, and store plain JSON so
 *      DuckDB can read it without unwrapping anything.
 *
 *   3. Fan out. R2 for the durable corpus; optionally a metrics vendor for
 *      dashboards. Adding a destination is one more call here — never a
 *      client release.
 *
 * What this deliberately does NOT do: log, store, or forward the source IP.
 * Cloudflare offers it as cf-connecting-ip; it is the one field that would
 * deanonymise install_id, so it is never read.
 */

// Mostly mirrors the payload built by _Session.payload(); an extension may
// also emit its own event carrying one of these top-level keys. A key absent
// here is dropped, not stored. Adding a metric means adding it here first —
// that friction is the point, and it is also the only privacy control that
// works retroactively, so it must land BEFORE any client starts sending the
// key or that traffic is silently discarded and unrecoverable.
const ALLOWED_KEYS = [
  'schema_version',
  'session',
  'tokens',
  'rates',
  'compression',
  'skips',
  'sources',
  'providers',
  'models',
  'failures',
  'failure_statuses',
  // Model-routing summary. Emitted by a routing extension rather than by the
  // proxy itself -- see proxy/route_advice.py for the decision seam. Same rule
  // as everything above: counters and model ids, no free text. Allowlisted
  // here so the corpus can answer what the proxy alone cannot -- a provider's
  // real minimum cacheable prefix, how long a cache actually survives, and how
  // far predicted cache hits are from the ones that happened.
  'routing',
];

// Resource attributes we keep. Same rule: allowlist, not denylist.
const ALLOWED_RESOURCE = [
  'service.name',
  'service.version',
  'headroom.install_id',
  'headroom.install_mode',
  'headroom.stack',
  'os.type',
  'host.arch',
];

// A beacon event is ~2KB. Anything far past that is a bug or an attack.
const MAX_BODY_BYTES = 64 * 1024;

/** OTLP AnyValue -> plain JS. The inverse of _any_value() in session.py. */
function unwrap(value) {
  if (value == null) return null;
  if ('stringValue' in value) return value.stringValue;
  if ('boolValue' in value) return value.boolValue;
  if ('intValue' in value) return Number(value.intValue);
  if ('doubleValue' in value) return value.doubleValue;
  if ('arrayValue' in value) return (value.arrayValue.values || []).map(unwrap);
  if ('kvlistValue' in value) {
    const out = {};
    for (const kv of value.kvlistValue.values || []) out[kv.key] = unwrap(kv.value);
    return out;
  }
  return null;
}

function pick(obj, allowed) {
  const out = {};
  if (!obj || typeof obj !== 'object') return out;
  for (const key of allowed) {
    if (key in obj) out[key] = obj[key];
  }
  return out;
}

/** OTLP ExportLogsServiceRequest -> flat, allowlisted records. */
function extract(payload) {
  const records = [];
  for (const rl of payload.resourceLogs || []) {
    const resource = {};
    for (const attr of rl.resource?.attributes || []) {
      resource[attr.key] = unwrap(attr.value);
    }
    const cleanResource = pick(resource, ALLOWED_RESOURCE);

    for (const sl of rl.scopeLogs || []) {
      for (const rec of sl.logRecords || []) {
        const body = unwrap(rec.body);
        if (!body || typeof body !== 'object') continue;
        records.push({
          ...pick(body, ALLOWED_KEYS),
          resource: cleanResource,
          // Server-stamped. A client clock can be wrong or forged; this is the
          // timestamp partitioning and retention actually rely on.
          received_at: new Date().toISOString(),
        });
      }
    }
  }
  return records;
}

// ----------------------------------------------------------------- rollup --
//
// The corpus is one object per heartbeat, ~1KB each — 65k on 2026-08-06 and
// climbing. DuckDB reads them correctly, but a full `pull` is ~100k HTTPS round
// trips for 95MB: minutes of pure per-object latency, no real bytes or compute.
// Listing the bucket alone took 88 seconds.
//
// This job collapses each COMPLETE hour into one object under rollup/, keeping
// only the highest-seq heartbeat per (install, session). One measured hour
// (dt=2026-08-06/hh=14): 3,938 objects and 3,938 rows in, 1 object and 1,061
// rows out. Analysis reads rollup/**, never sessions/**. Raw is left exactly as
// written, so any rollup can be rebuilt by deleting it.
//
// Hourly rather than daily because every R2 binding call is a subrequest: a day
// is ~65k of them against a 10k-per-invocation ceiling, an hour is ~4k.

const LOOKBACK_HOURS = 48; // heals a gap left by an outage or a deploy
const READ_BUDGET = 60000; // objects per run; see [limits] in wrangler.toml
// A get costs ~45ms of round trip and almost no CPU, so this is what decides
// whether a run finishes: at 20 an hour took ~3 minutes, against a 15-minute
// wall clock for a cron invocation. Raise it if an hour ever stops fitting.
const FANOUT = 100;        // concurrent R2 gets

const partition = (d) =>
  `dt=${d.toISOString().slice(0, 10)}/hh=${d.toISOString().slice(11, 13)}`;

/** One hour of heartbeats -> one deduped NDJSON object. Returns objects read. */
export async function rollupHour(env, part) {
  const best = new Map();
  let read = 0;
  let cursor;
  do {
    const page = await env.CORPUS.list({ prefix: `sessions/${part}/`, cursor });
    for (let i = 0; i < page.objects.length; i += FANOUT) {
      const texts = await Promise.all(
        page.objects
          .slice(i, i + FANOUT)
          .map((o) => env.CORPUS.get(o.key).then((r) => r?.text() ?? ''))
      );
      for (const text of texts) {
        read++;
        for (const line of text.split('\n')) {
          if (!line) continue;
          let rec;
          try {
            rec = JSON.parse(line);
          } catch {
            continue; // a single unreadable object must not lose the hour
          }
          // A session heartbeats every 5 minutes carrying CUMULATIVE totals, so
          // the highest seq IS the whole session and every earlier row is a
          // strict subset. Sessions straddle hours, so readers still dedupe
          // across rollups on this same key — this only shrinks each hour.
          const id = `${rec.resource?.['headroom.install_id']}\u0000${rec.session?.id}`;
          const prev = best.get(id);
          if (!prev || (rec.session?.seq ?? 0) > (prev.session?.seq ?? 0)) {
            best.set(id, rec);
          }
        }
      }
    }
    cursor = page.truncated ? page.cursor : undefined;
  } while (cursor);

  // An empty hour writes nothing rather than a zero-byte object every reader
  // would have to special-case. It stays "missing" and is retried until it
  // falls out of the lookback window, which costs one LIST.
  if (best.size) {
    await env.CORPUS.put(
      `rollup/${part}/data.ndjson`,
      [...best.values()].map((r) => JSON.stringify(r)).join('\n'),
      { httpMetadata: { contentType: 'application/x-ndjson' } }
    );
  }
  return read;
}

export default {
  /** Hourly cron. Builds every complete hour in the window that has no rollup. */
  async scheduled(event, env) {
    // ponytail: lists all of rollup/ each run — one request per 1000 hours of
    // history. Scope it to the window if that ever shows up in the bill.
    const done = new Set();
    let cursor;
    do {
      const page = await env.CORPUS.list({ prefix: 'rollup/', cursor });
      for (const o of page.objects) {
        done.add(o.key.slice('rollup/'.length, -'/data.ndjson'.length));
      }
      cursor = page.truncated ? page.cursor : undefined;
    } while (cursor);

    // Newest first, so a backlog drains from the present backwards and the
    // freshest hour is never the one starved by the budget. Starts at i=1: the
    // current hour is still being written to and is not a complete hour yet.
    let budget = READ_BUDGET;
    for (let i = 1; i <= LOOKBACK_HOURS && budget > 0; i++) {
      const part = partition(new Date(event.scheduledTime - i * 3600_000));
      if (done.has(part)) continue;
      try {
        budget -= await rollupHour(env, part);
      } catch (err) {
        // Newest-first means an hour that always throws — one grown past the
        // subrequest ceiling, say — would otherwise block every older hour
        // behind it forever. Skip it and keep draining; the next run retries
        // it while it is still inside the lookback window.
        console.error(`rollup ${part} failed: ${err}`);
        budget -= 1000; // unknown spend, so assume a page's worth
      }
    }
  },

  async fetch(request, env, ctx) {
    if (request.method !== 'POST') {
      return new Response('beacon: POST OTLP logs to /v1/logs', { status: 405 });
    }
    const url = new URL(request.url);
    if (url.pathname !== '/v1/logs') {
      return new Response('not found', { status: 404 });
    }

    const raw = await request.arrayBuffer();
    if (raw.byteLength > MAX_BODY_BYTES) {
      return new Response('payload too large', { status: 413 });
    }

    let records;
    try {
      records = extract(JSON.parse(new TextDecoder().decode(raw)));
    } catch {
      // Malformed input is not worth a retry storm from clients.
      return new Response('bad request', { status: 400 });
    }
    if (records.length === 0) return new Response(null, { status: 204 });

    const now = new Date();
    const day = now.toISOString().slice(0, 10);
    const hour = now.toISOString().slice(11, 13);
    // Hive-style partitioning so DuckDB can prune by date without a catalog.
    // ponytail: one object per request. Compacted hourly into rollup/ by
    // scheduled() below — analysis reads that, never this.
    const key = `sessions/dt=${day}/hh=${hour}/${crypto.randomUUID()}.json`;
    const ndjson = records.map((r) => JSON.stringify(r)).join('\n');

    // Respond immediately; durability work continues after the response.
    // The client is fire-and-forget and ignores the status anyway — making it
    // wait on R2 would only add latency to someone else's coding session.
    ctx.waitUntil(
      env.CORPUS.put(key, ndjson, {
        httpMetadata: { contentType: 'application/x-ndjson' },
      })
    );

    // Optional second lane: forward verbatim OTLP to a metrics backend for
    // dashboards. Configured by secret, so it can be added or swapped with a
    // `wrangler secret put` and no code change.
    if (env.METRICS_OTLP_URL) {
      ctx.waitUntil(
        fetch(env.METRICS_OTLP_URL, {
          method: 'POST',
          headers: {
            'content-type': 'application/json',
            authorization: env.METRICS_OTLP_AUTH || '',
          },
          body: JSON.stringify({ resourceLogs: [{ scopeLogs: [{ logRecords: records.map((r) => ({ body: { stringValue: JSON.stringify(r) } })) }] }] }),
        }).catch(() => {})
      );
    }

    return new Response(null, { status: 204 });
  },
};
