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

// uoft-cs/cifar10 (the canonical Hub mirror of the CIFAR-10 image
// classification dataset, from its own authors): 10 classes, 32x32 RGB.
// Row count is the known size of plain_text/train, confirmed live via the
// Hub's own dataset viewer (huggingface.co/datasets/uoft-cs/cifar10/viewer/
// plain_text/train) -- as is the image column name, `img` (checked below
// defensively anyway, in case a different config ever names it `image`).
//
// This is also, as of this writing, IMAGE_DATASET's source (see below) --
// a second, unrelated Hub dataset originally filled that role
// (frgfm/imagenette) until its own CI run found it completely broken: the
// Hub's modern dataset-server no longer serves datasets defined by a
// script (`imagenette.py`) at all, so *every* request to it 404s
// regardless of offset -- not a stale-row-count problem the retry in
// fetchRowsPage below can paper over, an entirely dead data source.
// CIFAR-10 was already confirmed working for fetchCifar10Batch, so
// IMAGE_DATASET was pointed at it too rather than hunting for a second
// still-working replacement. The one real cost: CIFAR-10's 32x32 source
// images are much smaller than Imagenette's own 160px ones, so a caller
// resizing up to a large target (a real vision model's actual input size,
// in the "Run inference" panel's own generic use) gets a softer result
// than before -- a real quality regression, accepted over a dead feature.
const IMAGE_DATASET = { dataset: "uoft-cs/cifar10", config: "plain_text", split: "train", rows: 50000 };
const IMAGE_LABELS = [
  "airplane", "automobile", "bird", "cat", "deer",
  "dog", "frog", "horse", "ship", "truck",
];

// stanfordnlp/sst2: short single-sentence movie-review snippets. Row count is
// the known size of default/validation.
const TEXT_DATASET = { dataset: "stanfordnlp/sst2", config: "default", split: "validation", rows: 872 };

function rowsUrl({ dataset, config, split }, offset, length = 1) {
  const p = new URLSearchParams({ dataset, config, split, offset: String(offset), length: String(length) });
  return `${ROWS_API}?${p.toString()}`;
}

// Fetch a page of rows at `offset`, retrying once at offset 0 on a 404. The
// `rows` counts above (IMAGE_DATASET/TEXT_DATASET) are hand-maintained
// estimates that can drift as a dataset/config/split is edited on the Hub --
// a 404 at a plausible-looking offset means the real row count is smaller
// than assumed, not that the dataset itself is gone (offset 0 is always
// valid for a dataset/config/split that resolves at all). Retrying there
// once turns a stale estimate into a less-random pick instead of a failed
// run. This does NOT help when the *whole dataset* is gone (as
// frgfm/imagenette turned out to be -- see IMAGE_DATASET's own comment):
// that 404s at every offset, retry included, and the caller sees the error
// from the offset-0 attempt.
async function fetchRowsPage(source, offset, length) {
  const url = rowsUrl(source, offset, length);
  const r = await fetch(url);
  if (r.ok) return r.json();
  if (r.status === 404 && offset !== 0) {
    const retryUrl = rowsUrl(source, 0, length);
    const retry = await fetch(retryUrl);
    if (retry.ok) return retry.json();
    throw new Error(
      `Hugging Face dataset viewer returned HTTP ${retry.status} for ${retryUrl} (after a 404 at offset ${offset})`,
    );
  }
  throw new Error(`Hugging Face dataset viewer returned HTTP ${r.status} for ${url}`);
}

// Fetch a single random row from a dataset config/split. Returns the row's
// `row` object (the dataset's own column shape) or throws on a network/HTTP
// failure.
async function fetchRandomRow(source) {
  const offset = Math.floor(Math.random() * Math.max(1, source.rows));
  const data = await fetchRowsPage(source, offset, 1);
  const row = data.rows && data.rows[0] && data.rows[0].row;
  if (!row) throw new Error(`no rows returned for ${source.dataset} (${source.config}/${source.split})`);
  return row;
}

// Fetch one random sample image from IMAGE_DATASET (uoft-cs/cifar10).
// Returns { bytes: Uint8Array, label: string } where `label` is the class
// name (e.g. "frog"), for a log line describing what was fed to the model.
export async function fetchSampleImageBytes() {
  const row = await fetchRandomRow(IMAGE_DATASET);
  const src = (row.img && row.img.src) || (row.image && row.image.src);
  const label = IMAGE_LABELS[row.label] || `class ${row.label}`;
  if (!src) throw new Error("sample row had neither img.src nor image.src");
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

// Fetch `numSamples` *consecutive* real rows from IMAGE_DATASET in one
// request (unlike fetchSampleImageBytes/fetchSampleSentence, which always
// fetch a single row -- a genuine "pretraining sample" needs several
// distinct labeled examples, and the rows API already supports a `length`
// greater than 1, so one call gets the whole batch rather than numSamples
// separate round trips). The offset is still randomized, so consecutive
// calls see a different slice of the dataset; the samples within one call
// are the dataset's own consecutive row order, not individually reshuffled.
//
// Returns an array of { bytes: Uint8Array, label: number, labelName: string },
// in row order. Throws on a network/HTTP failure, on a row with neither an
// `img` nor an `image` field with a `.src`, or if the response has fewer
// than `numSamples` rows (a truncated batch would silently train on less
// data than the caller asked for).
export async function fetchCifar10Batch(numSamples) {
  const offset = Math.floor(Math.random() * Math.max(1, IMAGE_DATASET.rows - numSamples));
  const data = await fetchRowsPage(IMAGE_DATASET, offset, numSamples);
  const rows = (data.rows || []).map((entry) => entry.row);
  if (rows.length < numSamples) {
    throw new Error(
      `requested ${numSamples} rows from ${IMAGE_DATASET.dataset} but got ${rows.length}`,
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
    samples.push({ bytes, label, labelName: IMAGE_LABELS[label] || `class ${label}` });
  }
  return samples;
}
