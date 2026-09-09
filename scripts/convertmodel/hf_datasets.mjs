// Fetch real sample inputs (an image, a sentence) from Hugging Face datasets
// for the "Run inference" panel's "sample data" fill mode — a real-looking
// counterpart to input_fill.mjs's synthetic random/ones/zeros/arange modes.
//
// Like hf_models.mjs, this module is pure data/networking: it returns bytes
// (or text) and a human-readable label, never touching the DOM. The Hub's
// public Dataset Viewer API (datasets-server.huggingface.co) serves both the
// row JSON and (for images) a short-lived signed asset URL with permissive
// CORS — the same endpoint HF's own embeddable dataset-viewer widget uses — so
// a browser `fetch` is all that's needed, mirroring how hf_models.mjs pulls
// model files straight from the Hub with no server-side proxy.
//
// The image URLs the rows API returns are signed and expire, so callers should
// use the bytes immediately rather than caching the URL itself; fetchSample*()
// always does a fresh row lookup, never memoizes a picked row.

const ROWS_API = "https://datasets-server.huggingface.co/rows";

// frgfm/imagenette (Apache-2.0): 10 easily-recognized ImageNet classes (tench,
// church, chainsaw, ...). The 160px config keeps the fetched JPEG small. Row
// count is the known size of 160px/validation (checked via the Hub dataset
// viewer) — only used to bound the random offset, so it doesn't need to track
// the dataset exactly.
const IMAGE_DATASET = { dataset: "frgfm/imagenette", config: "160px", split: "validation", rows: 3900 };

// stanfordnlp/sst2: short single-sentence movie-review snippets. Row count is
// the known size of default/validation.
const TEXT_DATASET = { dataset: "stanfordnlp/sst2", config: "default", split: "validation", rows: 872 };

// uoft-cs/cifar10 (the canonical Hub mirror of the CIFAR-10 image
// classification dataset, from its own authors): 10 classes, 32x32 RGB, no
// signed/expiring URL quirks beyond what frgfm/imagenette above already
// handles. Row count is the known size of plain_text/train.
//
// Unlike IMAGE_DATASET/TEXT_DATASET, this repo's own generated column name
// has not been checked against a live response (this dev sandbox's outbound
// network is proxy-blocked to huggingface.co -- see
// webgpu_cifar10_pretrain.test.mjs's own top comment). Parquet-converted
// image datasets on the Hub name their image column after the original
// dataset's own field, which for CIFAR-10 uploads is `img` far more often
// than `image` -- but fetchCifar10Batch checks both defensively rather than
// assume, so a wrong guess here is a clear thrown error (naming the row's
// own keys) on the first live CI run, not a silent empty batch.
const CIFAR10_DATASET = { dataset: "uoft-cs/cifar10", config: "plain_text", split: "train", rows: 50000 };
const CIFAR10_LABELS = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

function rowsUrl({ dataset, config, split }, offset, length = 1) {
  const p = new URLSearchParams({ dataset, config, split, offset: String(offset), length: String(length) });
  return `${ROWS_API}?${p.toString()}`;
}

// Fetch a single random row from a dataset config/split. Returns the row's
// `row` object (the dataset's own column shape) or throws on a network/HTTP
// failure.
async function fetchRandomRow(source) {
  const offset = Math.floor(Math.random() * Math.max(1, source.rows));
  const url = rowsUrl(source, offset, 1);
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`Hugging Face dataset viewer returned HTTP ${r.status} for ${url}`);
  }
  const data = await r.json();
  const row = data.rows && data.rows[0] && data.rows[0].row;
  if (!row) throw new Error(`no rows returned for ${source.dataset} (${source.config}/${source.split})`);
  return row;
}

// Fetch one random sample image from frgfm/imagenette. Returns
// { bytes: Uint8Array, label: string } where `label` is the class name (e.g.
// "church"), for a log line describing what was fed to the model.
export async function fetchSampleImageBytes() {
  const row = await fetchRandomRow(IMAGE_DATASET);
  const src = row.image && row.image.src;
  const labelNames = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
  ];
  const label = labelNames[row.label] || `class ${row.label}`;
  if (!src) throw new Error("sample row had no image.src");
  const imgResp = await fetch(src);
  if (!imgResp.ok) {
    throw new Error(`failed to download sample image: HTTP ${imgResp.status}`);
  }
  const bytes = new Uint8Array(await imgResp.arrayBuffer());
  return { bytes, label };
}

// Fetch one random sample sentence from stanfordnlp/sst2. Returns
// { text: string, label: string } where `label` is "positive"/"negative".
export async function fetchSampleSentence() {
  const row = await fetchRandomRow(TEXT_DATASET);
  const text = row.sentence;
  if (!text) throw new Error("sample row had no sentence field");
  const label = row.label === 1 ? "positive" : row.label === 0 ? "negative" : `label ${row.label}`;
  return { text, label };
}

// Fetch `numSamples` *consecutive* real CIFAR-10 rows in one request (unlike
// fetchSampleImageBytes/fetchSampleSentence, which always fetch a single row
// -- a genuine "pretraining sample" needs several distinct labeled examples,
// and the rows API already supports a `length` greater than 1, so one call
// gets the whole batch rather than numSamples separate round trips). The
// offset is still randomized, so consecutive calls see a different slice of
// the dataset; the samples within one call are the dataset's own consecutive
// row order, not individually reshuffled.
//
// Returns an array of { bytes: Uint8Array, label: number, labelName: string },
// in row order. Throws on a network/HTTP failure, on a row with neither an
// `img` nor an `image` field with a `.src` (see CIFAR10_DATASET's own
// comment on why both are checked), or if the response has fewer than
// `numSamples` rows (a truncated batch would silently train on less data
// than the caller asked for).
export async function fetchCifar10Batch(numSamples) {
  const offset = Math.floor(Math.random() * Math.max(1, CIFAR10_DATASET.rows - numSamples));
  const url = rowsUrl(CIFAR10_DATASET, offset, numSamples);
  const r = await fetch(url);
  if (!r.ok) {
    throw new Error(`Hugging Face dataset viewer returned HTTP ${r.status} for ${url}`);
  }
  const data = await r.json();
  const rows = (data.rows || []).map((entry) => entry.row);
  if (rows.length < numSamples) {
    throw new Error(
      `requested ${numSamples} rows from ${CIFAR10_DATASET.dataset} but got ${rows.length}`,
    );
  }

  const samples = [];
  for (const row of rows) {
    const src = (row.img && row.img.src) || (row.image && row.image.src);
    if (!src) {
      throw new Error(
        `sample row had neither img.src nor image.src -- row keys: ${Object.keys(row).join(", ")}`,
      );
    }
    const imgResp = await fetch(src);
    if (!imgResp.ok) {
      throw new Error(`failed to download sample image: HTTP ${imgResp.status}`);
    }
    const bytes = new Uint8Array(await imgResp.arrayBuffer());
    const label = row.label;
    samples.push({ bytes, label, labelName: CIFAR10_LABELS[label] || `class ${label}` });
  }
  return samples;
}
