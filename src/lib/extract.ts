export type Phrase = { text: string; norm: string; count: number };

export type Feature = {
  feature_num: number;
  case_num: number;
  feature_text: string;
  display: string;
  phrases: Phrase[];
};

export type Entity = {
  start: number;
  end: number;
  text: string;
  type: string;
  feature_num: number;
  feature_text: string;
  confidence: number;
  source: "lexicon" | "keyword";
};

const STOP = new Set(["or", "of", "no", "to", "a", "the", "and", "with", "in", "for", "on", "at"]);
const BANNED = new Set([...STOP, "old", "st", "ing", "er", "ed", "ly", "al", "an", "as", "is", "be", "by"]);

function keepPhrase(phrase: string, featureText: string): boolean {
  const p = phrase.trim();
  if (!p) return false;
  if (/^[,./;:\-–—()[\]{}"'`]+$/.test(p)) return false;
  const low = p.toLowerCase();
  if (BANNED.has(low)) return false;
  const feat = featureText.toLowerCase();
  if (p.length === 1) return "MF".includes(p.toUpperCase()) && (feat === "male" || feat === "female");
  if (p.length === 2) return ["yo", "hx", "mi", "ed", "iud"].includes(low);
  return p.length >= 3;
}

function isBoundary(text: string, start: number, end: number): boolean {
  const left = start === 0 || !/[A-Za-z0-9]/.test(text[start - 1] ?? "");
  const right = end >= text.length || !/[A-Za-z0-9]/.test(text[end] ?? "");
  return left && right;
}

function keywords(featureText: string): string[] {
  return featureText
    .toLowerCase()
    .replace(/-/g, " ")
    .split(/\s+/)
    .filter((t) => t.length > 2 && !STOP.has(t));
}

function overlaps(a: Entity, b: Entity): boolean {
  return !(a.end <= b.start || b.end <= a.start);
}

function findAll(haystackLower: string, needle: string): number[] {
  const hits: number[] = [];
  if (needle.length < 2) return hits;
  let from = 0;
  while (from <= haystackLower.length - needle.length) {
    const idx = haystackLower.indexOf(needle, from);
    if (idx < 0) break;
    hits.push(idx);
    from = idx + Math.max(1, needle.length);
  }
  return hits;
}

export function extractNote(text: string, features: Feature[]): Entity[] {
  if (!text.trim() || features.length === 0) return [];
  const lower = text.toLowerCase();
  const out: Entity[] = [];

  for (const feat of features) {
    const found: Entity[] = [];
    const phrases = [...feat.phrases].sort((a, b) => b.text.length - a.text.length);
    const maxCount = Math.max(1, ...phrases.map((p) => p.count));

    for (const phrase of phrases) {
      const needle = phrase.text.toLowerCase().trim();
      if (!keepPhrase(phrase.text, feat.feature_text)) continue;
      for (const start of findAll(lower, needle)) {
        const end = start + needle.length;
        if (needle.length <= 4 && !isBoundary(text, start, end)) continue;
        const entity: Entity = {
          start,
          end,
          text: text.slice(start, end),
          type: feat.display,
          feature_num: feat.feature_num,
          feature_text: feat.feature_text,
          confidence: Math.min(0.97, 0.55 + 0.4 * (phrase.count / maxCount)),
          source: "lexicon",
        };
        if (found.some((e) => overlaps(e, entity))) continue;
        found.push(entity);
      }
    }

    const isNegation = feat.display.toLowerCase().startsWith("no ") || feat.feature_text.toLowerCase().startsWith("No-") || feat.feature_text.toLowerCase().startsWith("no-");
    if (found.length === 0 && !isNegation) {
      const kws = keywords(feat.feature_text).filter((k) => k !== "year");
      if (kws.length) {
        const window = 48;
        for (const kw of kws) {
          for (const idx of findAll(lower, kw)) {
            const start = Math.max(0, idx);
            const end = Math.min(text.length, idx + kw.length);
            // Expand to nearby words if other keywords sit close.
            let lo = start;
            let hi = end;
            let hits = 1;
            for (const other of kws) {
              if (other === kw) continue;
              const region = lower.slice(Math.max(0, start - window), Math.min(lower.length, end + window));
              if (region.includes(other)) hits += 1;
            }
            if (hits < Math.min(2, kws.length) && kws.length > 1) continue;
            const entity: Entity = {
              start: lo,
              end: hi,
              text: text.slice(lo, hi),
              type: feat.display,
              feature_num: feat.feature_num,
              feature_text: feat.feature_text,
              confidence: 0.42 + Math.min(0.2, hits * 0.06),
              source: "keyword",
            };
            if (found.some((e) => overlaps(e, entity))) continue;
            found.push(entity);
          }
        }
      }
    }

    found.sort((a, b) => b.confidence - a.confidence || b.end - b.start - (a.end - a.start));
    const kept: Entity[] = [];
    for (const e of found) {
      if (kept.some((k) => overlaps(k, e))) continue;
      kept.push(e);
      if (kept.length >= 4) break;
    }
    out.push(...kept);
  }

  return out.sort((a, b) => a.start - b.start || a.end - b.end);
}

export function scoreCase(text: string, features: Feature[]): number {
  if (!text.trim()) return 0;
  const lower = text.toLowerCase();
  let score = 0;
  for (const feat of features) {
    for (const p of feat.phrases.slice(0, 12)) {
      if (p.text.length >= 4 && lower.includes(p.text.toLowerCase())) {
        score += 2 + Math.min(3, p.count / 5);
        break;
      }
    }
  }
  return score;
}

export function detectCase(text: string, all: Feature[], profiles: { case_num: number }[]): number {
  let best = profiles[0]?.case_num ?? 0;
  let bestScore = -1;
  for (const p of profiles) {
    const feats = all.filter((f) => f.case_num === p.case_num);
    const s = scoreCase(text, feats);
    if (s > bestScore) {
      bestScore = s;
      best = p.case_num;
    }
  }
  return best;
}

export type Segment = { text: string; entity?: Entity; key: string };

export function segmentText(text: string, entities: Entity[]): Segment[] {
  if (!text) return [];
  const sorted = [...entities].sort((a, b) => a.start - b.start || b.end - a.end);
  const picked: Entity[] = [];
  for (const e of sorted) {
    if (e.start < 0 || e.end > text.length || e.end <= e.start) continue;
    if (picked.some((p) => !(e.end <= p.start || e.start >= p.end))) continue;
    picked.push(e);
  }
  picked.sort((a, b) => a.start - b.start);
  const segs: Segment[] = [];
  let cursor = 0;
  picked.forEach((e, i) => {
    if (e.start > cursor) {
      segs.push({ text: text.slice(cursor, e.start), key: `t-${cursor}` });
    }
    segs.push({ text: text.slice(e.start, e.end), entity: e, key: `e-${i}-${e.start}` });
    cursor = e.end;
  });
  if (cursor < text.length) segs.push({ text: text.slice(cursor), key: `t-${cursor}` });
  return segs;
}

const TINTS = [
  "bg-highlight text-fg",
  "bg-highlight-2 text-fg",
  "bg-highlight-3 text-fg",
];

export function tintFor(featureNum: number): string {
  return TINTS[Math.abs(featureNum) % TINTS.length] ?? TINTS[0];
}
