# Photo search guide

Natural-language search over a personal photo library, entirely on your own machine. This is the
user guide. For how it is built and why, see [search-design.md](search-design.md); for the short
version, see the [README](../README.md#photo-search).

Contents: [Concepts](#concepts) · [Install](#install) · [Quick start](#quick-start) ·
[Commands](#commands) · [Dates](#dates) · [Places](#places) · [People](#people) ·
[Query parsing modes](#query-parsing-modes) · [Models and downloads](#models-and-downloads) ·
[The index](#the-index) · [Privacy](#privacy) · [Evaluation](#evaluation) ·
[Limitations](#limitations) · [Troubleshooting](#troubleshooting) · [Exit codes](#exit-codes) ·
[Licences](#licences)

## Concepts

**The index.** `facesorter index` reads every still photo in a folder once and stores, in one
SQLite file: the capture date, the GPS position and a city/region/country label derived from it,
the camera, and an embedding (a 768-number vector) of what the picture looks like. Searching never
touches the photos themselves.

**Two kinds of query content.** A query such as `beach at sunset in Lisbon last summer` is split
into:

- **Filters**, applied exactly: a date range (`last summer`), a place (`Lisbon`) and named people.
  A photo either satisfies a filter or it does not.
- **Visual text** (`beach at sunset`): ranked by an image-text model (SigLIP 2). Every photo that
  passes the filters is scored by how well its embedding matches the text, and the best `k` are
  returned.

A query with only filters (`July 2023`) has no visual text; the matching photos are listed newest
first and the score column is 0.

**Grounding.** Whatever reads your query (rules or a small language model) can only pick out words
that really appear in it. A made-up filter therefore cannot silently hide photos. The visual text
is rebuilt from your own words with the filter words removed.

**Ask once, then tell.** If part of a query looks like a date, place or person but cannot be turned
into a filter (for example `since 2020`, or a name nobody has been given), the CLI asks you once to
reword it when it is run from a terminal. Press Enter to skip. Anything still not understood, and
everything when `--no-ask` is used or input is not a terminal, is reported
(`I couldn't understand "since 2020" as a date, so I searched without it.`) and the search runs
without that filter. See [Dates](#dates) for what is deliberately not supported.

## Install

Photo search needs **Python 3.10 or newer**. The core `sort` and `find` commands need only
Python 3.9 or newer.

```bash
pip install "facesorter[search]"
```

The `search` extra installs PyTorch and open_clip (image embeddings at index time), onnxruntime
and `tokenizers` (text embeddings at query time, no PyTorch needed), `geonamescache` and `scipy`
(offline places), `dateparser`, `huggingface_hub` and Gradio (the local web app). Two optional
extras add a local language model for query understanding; install at most one:

| Extra | Runtime | Needs |
| --- | --- | --- |
| `facesorter[search-mlx]` | MLX (`mlx-lm`) | Apple-silicon Mac and Python 3.11 or newer |
| `facesorter[search-llama]` | `llama-cpp-python` (GGUF models) | Any OS, Python 3.10 or newer; may build from source on some platforms |

With neither, queries are understood by deterministic rules ([modes](#query-parsing-modes)); that
is a supported mode, not an error. `facesorter[search-core]` is the light subset (dates, places,
tokenizer, no model runtimes) that CI installs.

On Python 3.9 the search extras install nothing (their dependencies are marked Python 3.10+), so
`pip install "facesorter[search]"` succeeds but adds no search stack. `facesorter --help` and
`sort`/`find` work, and the search commands exit with code 2 and a message saying search needs
Python 3.10 or newer. CI runs the whole test suite on 3.9 to 3.13 with `.[dev,search-core]`; the
search tests that need a missing package skip cleanly on 3.9.

## Quick start

```bash
# 1. Index a folder (first run downloads the image model; see "Models and downloads")
facesorter index ~/Pictures

# 2. Search in natural language
facesorter search "beach at sunset in Lisbon last summer"

# or with explicit filters, which override anything found in the query text
facesorter search "sunset" --place Lisbon --from 2023-06 --to 2023-08 -k 20

# 3. The local web app (http://127.0.0.1:7860)
facesorter serve

# 4. Optional: find people, then name them so "with Bob" works
facesorter people scan
facesorter people cluster
facesorter people list
facesorter people name 3 Bob
facesorter search "Bob at the beach"
```

Re-running `facesorter index ~/Pictures` after adding photos only processes the new and changed
files. A synthetic example of what a search prints (made from generated test images and the test
fakes, so the scores are meaningless):

```text
$ facesorter search "orange photos in Lisbon July 2023" --no-llm --no-ask
Searching: dates 2023-07-01 to 2023-07-31 · place Lisbon, Portugal
#   score  date        place             path
1   0.999  2023-07-15  Lisbon, Portugal  /home/me/Pictures/b.jpg
2  -0.534  2023-07-14  Lisbon, Portugal  /home/me/Pictures/a.jpg
```

The line starting `Searching:` states which filters were applied. Notices (what was not
understood, what was dropped) go to stderr.

## Commands

All commands accept `-v/--verbose` (also before the subcommand) and `--index PATH` to use a
different index file ([location](#the-index)). `eval` has its own option list, below.

### `facesorter index [LIBRARY]`

Scan `LIBRARY` recursively for `.jpg`, `.jpeg`, `.png`, `.heic` and `.heif` files (any letter
case), read their metadata, embed their pixels and store everything. Hidden files and folders,
macOS `._*` files, videos (including Live Photo `.mov` companions) and symlinked folders are
skipped. Photos are only ever read.

| Option | Default | Description |
| --- | --- | --- |
| `LIBRARY` | required, unless `--remove-root` is given | Folder of photos to index |
| `--workers N` | `4` | Decoder threads |
| `--batch-size N` | `32` | Images per model batch |
| `--limit N` | all | Index at most N new or changed photos (a trial run); a limited run never removes anything from the index |
| `--force` | off | Re-embed photos even if unchanged |
| `--exclude DIR` | none | Skip this folder; repeatable |
| `--device {auto,mps,cuda,cpu}` | `auto` | Compute device for the image model: `auto` picks the Apple GPU (`mps`), then CUDA, then CPU. Half precision is used on `mps`/`cuda`, full precision on the CPU |
| `--allow-empty-prune` | off | Allow removing every photo of a library from the index when the scan finds none (see [lifecycle](#lifecycle-and-caveats)) |
| `--remove-root DIR` | none | Instead of indexing, forget the library folder `DIR`: its photos leave the index and no file is touched. `DIR` need not exist any more. Cannot be combined with `LIBRARY`; a folder that is not an indexed library is an error (exit 2), see [lifecycle](#lifecycle-and-caveats) |
| `--index PATH` | [default](#the-index) | Index file |

It prints a summary (`scanned`, `new`, `updated`, `unchanged`, `removed`, `failed`, time, model
name). Photos that could not be read are counted under `failed` and retried next time; add `-v` to
list them (paths are not printed otherwise). Progress bars go to stderr. Ctrl-C keeps every
finished chunk of 256 photos, and the next run continues.

### `facesorter search [QUERY ...]`

| Option | Default | Description |
| --- | --- | --- |
| `QUERY` | none | What to look for; several words need no quoting |
| `--from DATE`, `--to DATE` | none | `YYYY-MM-DD`, `YYYY-MM` or `YYYY`. `--from 2024-02` starts on 2024-02-01, `--to 2024-02` ends on 2024-02-29 (leap years respected), `--to 2023` ends on 2023-12-31. Either may be given alone |
| `--place NAME` | none | Only photos taken at this place; a city, region, country or [known destination](#places) |
| `--person NAME` | none | A named person ([People](#people)); repeatable, and every named person must be in the photo |
| `-k N` | `10` | Number of results |
| `--no-llm` | off | Understand the query with rules only |
| `--llm {auto,mlx,llama-cpp,none}` | `auto` | Local language-model backend ([modes](#query-parsing-modes)) |
| `--llm-model MODEL` | backend default | Model name or path |
| `--no-ask` | off | Never ask follow-up questions; report and search without |
| `--json` | off | One JSON object per result line (`rank`, `score`, `id`, `path`, `date`, `place`) |
| `--paths-only` | off | Print only the file paths. Mutually exclusive with `--json` |
| `--copy-to DIR` | none | Copy the result photos into this folder (never overwrites; existing names get `_1`, `_2`, ...) |
| `--index PATH` | [default](#the-index) | Index file |

Give a query, or at least one of `--from`, `--to`, `--place`, `--person`; otherwise it exits with
code 2. Explicit flags win over what the parser finds in the query, field by field. A flag that
cannot be used is an error (exit 2), not a notice: an impossible date, a `--place` the gazetteer
does not know, a `--person` nobody has been named. Scores are cosine similarities between the
query text and the photo; compare them within one query only. A search with no results prints
`No matching photos.` and still exits 0, with a hint about the dates or places the library does
cover when a date or place filter alone matches nothing.

With `--json` or `--paths-only` the `Searching:` line moves to stderr too, so stdout holds only results and can be piped.

### `facesorter serve`

Starts the local web app and blocks until Ctrl-C.

| Option | Default | Description |
| --- | --- | --- |
| `--host HOST` | `127.0.0.1` | Address to bind |
| `--port PORT` | `7860` | Port |
| `--allow-remote` | off | Allow a non-loopback `--host`. Without it, any other host is refused with exit code 2 |
| `--open-browser` | off | Open the page in a browser |
| `--no-llm`, `--llm`, `--llm-model` | as for `search` | Query-understanding backend |
| `--index PATH` | [default](#the-index) | Index file |

The page has a query box, optional date, place, people and result-count fields (which override the
query), the reading of your query ("How I read your query"), the clarification questions, a
gallery of thumbnails (24 results by default, up to 200), and thumbs up/down buttons. The ratings
go into the index and feed [`eval`](#evaluation). It needs an index that records the folders it
covers, so run `facesorter index` first.

### `facesorter people`

Face recognition for naming people. It uses the same InsightFace model as `sort` and `find`
(downloaded on first use, non-commercial licence, see [Licences](#licences)). All subcommands
accept `-v` and `--index`.

| Subcommand | Options | Description |
| --- | --- | --- |
| `people scan` | `--confidence 0.8` (0 to 1), `--limit N` | Detect faces in indexed photos that have not been scanned (or whose file changed). Photos larger than 1600 px on the long side are shrunk first for speed. Safe to interrupt: work is saved photo by photo |
| `people cluster` | `--threshold 0.6` | Group all stored faces into clusters (cosine distance, same method as `sort`). Names you gave are carried over |
| `people list` | | Table of `id`, `faces`, `photos` and `name` per cluster; with `-v`, three sample paths per cluster |
| `people name CLUSTER_ID [NAME]` | `--clear` | Name a cluster, or remove its name with `--clear` |

### `facesorter stats`

Shows what the index contains: file, model name, photo count, how many have GPS and a place label,
the date span, how many photos got their date from `exif`, `folder`, `mtime` or nothing, and the
number of faces, named people and stored ratings. Check the date-source line when date filters
seem to miss photos ([Dates](#dates)).

### `facesorter eval`

Caption-free quality report; see [Evaluation](#evaluation). Options:

| Option | Default | Description |
| --- | --- | --- |
| `--index PATH` | [default](#the-index) | Index file |
| `--sample N` | `200` | Photos to sample for the probes |
| `-k N` | `24` | Results per query |
| `--seed N` | `0` | Sampling seed |
| `--queries FILE` | none | JSON list of extra probes: `[{"query": "...", "months": [...], "places": [...], "countries": [...]}]` (`months`, `places`, `countries` optional) |
| `--json` | off | Print the report as JSON instead of Markdown |
| `--markdown FILE` | none | Also write the Markdown report to a file |
| `--show-paths` | off | Include file paths and feedback query text (default: aggregates only) |
| `--llm {auto,mlx,llama-cpp,none}`, `--no-llm` | `auto` | Backend for the probes; `--no-llm` is `--llm none` |

## Dates

Dates matter most for correctness, so the date reader is a small deterministic grammar, not a
guess. For a flag, use `YYYY`, `YYYY-MM` or `YYYY-MM-DD`. In a natural-language query these forms
are understood (English only). "Today" is the day you run the search; examples use 2026-11-09.

| You write | Means |
| --- | --- |
| `2019`, `July 2023`, `in July` | the whole year or month; a bare month is the most recent one that has started (July 2026) |
| `2023-07-14`, `14 July 2023`, `July 14, 2023`, `March 5th` | that day |
| `summer`, `last summer`, `fall 2021`, `winter 2020` | meteorological seasons: spring Mar-May, summer Jun-Aug, autumn/fall Sep-Nov, winter Dec-Feb. A winter is labelled by the year it ends in (`winter 2020` is Dec 2019 to Feb 2020). `last <season>` is the most recent completed one; a bare season may still be in progress |
| `Christmas 2022`, `Easter 2024`, `Thanksgiving`, `Halloween`, `Memorial Day`, `Labor Day`, `Valentine's Day`, `Fourth of July` | United States conventions; Christmas, Halloween, Easter and Thanksgiving get a window of one day either side; Memorial and Labor Day are the Saturday-to-Monday long weekend |
| `yesterday`, `last weekend`, `last month`, `this year`, `last Monday` | calendar periods, weeks Monday to Sunday |
| `last 30 days`, `past 6 months` | a rolling window ending today |
| `3 years ago`, `two weeks ago` | the calendar year, ISO week, month or day containing that moment |
| `between 2019 and 2021`, `June 2020 to August 2021`, `from June to August 2022` | one closed range |

Seasons assume the northern hemisphere. Times of day ("yesterday evening") are ignored: the
result is the whole day.

Deliberately **not** supported, because each would need a different kind of filter or would be a
guess. The search never reduces them to something else; you are asked (or told) instead:

- Open-ended or approximate wording: `since 2020`, `before June`, `after 2019`, `early 2020`,
  `mid 2020`, `late summer`. Use `--from`/`--to` or give both ends (`2020 to 2026`).
- Exclusions and alternatives: `not in summer`, `except 2019`, `2019 or 2020`, and a list of
  separate dates. Only one closed range can be applied.
- Ambiguous numeric dates such as `3/4/2024` or `03.04.2019` (day/month or month/day). Write
  `2024-03-04`.
- Lunar and religious-calendar holidays (Diwali, Eid, Lunar New Year, Hanukkah, ...). They are
  recognised and reported as not understood rather than guessed.
- Decades and quarters (`the 90s`, `Q3 2022`), other date vocabularies, and non-English date words
  (`été 2019`, `Weihnachten`).

When a phrase like these is found, the CLI asks you to reword it; if you skip, or `--no-ask` is
set, it prints a line such as `I couldn't understand "since 2020" as a date, so I searched
without it.`

Two things about free-text queries are worth knowing. First, to avoid claiming ordinary words
("may", "march", "Christmas lights", "beach summer"), the rules reader takes a bare month, season or
holiday as a date only with a lead word ("in July", "last summer", "this Christmas") or a year
("Christmas 2021"); `Christmas lights` stays visual text. Second, if you want to be certain, use
`--from` and `--to`.

**Which date does a photo have?** In order: the EXIF capture time (`DateTimeOriginal`, then
`DateTimeDigitized`, then `DateTime`); else a date in the folder names under the library root; else
the file's modification time. Only the first two are used by date filters, because a modification
time is usually the day you copied the file, not the day you took it. A photo with only a
modification time is still searchable by its picture but never matches a date filter (the `Date
source` line of `facesorter stats` tells you how many are in that state). Folder names recognised
(outermost folder wins ties, most specific date wins): `2023-07-14` (also `_`/`.` separators),
`2023-07`, a year folder followed by a month folder (`2023/07`, `2023/07 - July`, `2023/July`) and
a bare year folder (`2019`). A date taken from a folder name carries only month or year precision,
and such a photo matches any date range that overlaps that month or year, not one arbitrary day.
`search` shows it as `2023-07` or `2019` in the date column.

## Places

Everything about places is offline. The gazetteer comes from the `geonamescache` package (GeoNames,
CC BY 4.0, "Place data (c) GeoNames"); nothing is downloaded and no place data is bundled with
facesorter.

**Labelling.** At index time a photo with a GPS position gets one label: the nearest plausible
city (population of at least 15,000, not a neighbourhood of a bigger city), else just a country (and
region if they agree) for remote spots, else none for the open sea. Neighbourhoods are folded into
their city (Manhattan into New York City), while similarly sized neighbours stay separate. A photo
without GPS has no place and never matches a place filter; folder names are not used for places.

**Query names.** Accepted forms, case-, accent- and punctuation-insensitive (`Zurich` equals `Zürich`):

- a city: `Lisbon`; with a qualifier: `Paris, Texas`, `Amherst, MA`, `Vancouver, BC`
- a region: US states and DC, Canadian provinces, Australian states, the nations of the United
  Kingdom, Indian states, German states, Italian regions, French regions, Spanish autonomous
  communities and Austrian states. Other countries have country and city names only
  (`Hokkaido` is not understood unless it is one of the curated destinations).
- a country, by name or alias (`UK`, `Holland`), or by ISO code written in capitals (`FR`, `USA`)
- city aliases and former names (`NYC`, `SF`, `Bombay`)
- a curated list of 81 destinations that are not cities (`Cornwall`, `Bali`, `Lake Tahoe`,
  `Yosemite`, `Eiffel Tower`, ...). These match photos by distance from a point rather than by
  label, and so do real places too small to be labels (`Whistler`).

**Ambiguous names.** The gazetteer ranks candidates by a fixed rule: curated destinations and
aliases, then countries, then regions, then the biggest city (`Georgia` is the country,
`Washington` the state, `Paris` is Paris, France, `Amherst` is Amherst, New York). In the CLI and
the web app that ranking is library-aware: when a bare name has several readings and some have
photos in your index, the reading with the most photos wins (`Amherst` finds your Amherst,
Massachusetts photos if that is where you took them). A qualifier (`Amherst, MA`) always wins. If
no reading has photos, the gazetteer's own ranking stands.

**Known limits.** Cities within about 2 km across a river or border can land on the wrong side; some
suburbs of megacities merge into them (Guarulhos into São Paulo); label data comes from the
installed `geonamescache` release and can differ between versions; unknown names are not guessed;
only one place per query can be applied (`Lisbon and Porto` applies Lisbon and leaves "Porto"
in the visual text); and a place word is only picked out of a free-text query by the rules reader
when it is capitalised (`lisbon` stays visual text; use `--place` or an LLM). A name that is not
recognised in rules mode silently stays part of the visual text. In LLM mode an unknown place or
person is put to you as a question.

## People

People search needs two steps first, both done by you and both repeatable.

1. `facesorter people scan` finds faces in the photos in the index. It needs InsightFace's
   `buffalo_l` model, which InsightFace downloads from GitHub on first use (the same download
   `sort` and `find` trigger); this is the one network access of this step.
2. `facesorter people cluster` groups the faces into clusters, with the same cosine-distance DBSCAN
   (`min_samples=1`) as `facesorter sort`. A cluster is only a candidate for one person.

Then `facesorter people list` shows the clusters (largest first), `facesorter people name ID NAME`
names one, and `with NAME`, `--person NAME` or the web app's People box filter on it. Names are
unique ignoring case and accents. A photo matches `Bob` when it contains a face from the cluster
named Bob.

Things to know:

- **Names carry across re-clusters.** Run `people scan` and `people cluster` again after indexing
  more photos; clusters keep their ids where possible and each name moves to the new cluster that
  shares the most faces with its old cluster. A name is only passed to a cluster when it keeps
  enough of the old faces; names that cannot be carried over are dropped and listed in a warning.
- **Single-linkage chaining.** If face A is near B and B is near C, all three share a cluster even when
  A and C are far apart, so one blurry or look-alike face can merge two people. Lower
  `--threshold` (for example `0.5`) to split them, at the cost of splitting one person into several
  clusters. If one person is spread over several clusters, a name can only go on one of them.
- **10,000-face cap.** Clustering builds a full pairwise distance matrix, so memory grows with the
  square of the number of faces (about 1.3 GB at 10,000 faces). More than 10,000 faces is refused
  with an error. Raising `--threshold` does not help; scan fewer photos (`--limit`) or raise
  `--confidence` on `people scan`.
- **Re-indexing resets faces.** A photo whose file changed, or that was renamed or moved, is a new
  photo: its faces and ratings are dropped and it must be scanned again.
- No accuracy figures are claimed for face grouping; see the limitations in the README.

## Query parsing modes

`--llm` chooses what reads the query. All modes go through the same checks (grounding, date and
place resolution, asking or reporting).

| Mode | Behaviour |
| --- | --- |
| `auto` (default) | MLX on an Apple-silicon Mac when `mlx-lm` is installed, else llama.cpp when `llama-cpp-python` is installed, else rules. Nothing is installed or downloaded for you; with no runtime it quietly uses rules (add `-v` to see the log line) |
| `mlx` | MLX only; an error (exit 2) saying what to install if it cannot run |
| `llama-cpp` | llama.cpp only; same |
| `none` (or `--no-llm`) | Rules only |

The environment variables `FACESORTER_LLM_BACKEND` (`auto`, `mlx`, `llama-cpp`, `none`; replaces
`auto` only) and `FACESORTER_LLM_MODEL` set the defaults; command-line arguments win.

**Rules.** Dates come from the date grammar above. A place is a capitalised span, optionally with
`in`/`near`/`at`, that the gazetteer resolves; people are known names from `people name`. Anything
unresolved stays in the visual text, except date-like input, which is always asked about or
reported. Rules mode is fast (no model to load) and predictable.

**Local language model.** The model is asked for `{text, place, people, date_phrase}` as JSON, and
only to **copy** the date phrase out of your query, never to work out a date; code resolves it.
Everything it returns is grounded in the query, repaired from the rules where it left a field
empty or invented one, and an unknown place or name becomes a question. If the model fails or
returns unusable JSON, the rules are used and a notice says so. A query longer than 600
characters skips the model. The model helps most with free word order, lowercase place names and
names written in ways the rules do not catch.

| Backend | Default model | Download | Notes |
| --- | --- | --- | --- |
| MLX | `mlx-community/Qwen2.5-1.5B-Instruct-4bit` | about 0.9 GB | measured at roughly 1.3 s per query and 1.4 GB of memory on an Apple-silicon Mac |
| llama.cpp | `Qwen/Qwen2.5-1.5B-Instruct-GGUF`, file `qwen2.5-1.5b-instruct-q4_k_m.gguf` | about 1.1 GB | greedy decoding, any OS |

`--llm-model` takes, for MLX, a Hugging Face repo id or local folder; for llama.cpp a `.gguf` path,
`repo_id:filename`, or a repo id (the Q4_K_M file is chosen). The model loads on the first query:
each `facesorter search` is a new process and pays this, while `facesorter serve` keeps it loaded.
For the fastest one-shot searches use `--no-llm`.

## Models and downloads

Nothing is downloaded until you first need it, and the files are cached by `huggingface_hub`
(`~/.cache/huggingface` by default; `HF_HOME` moves it). Set `HF_HUB_OFFLINE=1` to forbid
downloads once everything is cached.

| Model | Used by | Size | Licence |
| --- | --- | --- | --- |
| SigLIP 2 B/16 (`ViT-B-16-SigLIP2`, `webli` weights via open_clip) | `index`: image embeddings | about 1.5 GB | Apache-2.0 |
| SigLIP 2 text tower, ONNX (`onnx-community/siglip2-base-patch16-224-ONNX`, `text_model_fp16.onnx` plus `tokenizer.json`) | `search`, `serve`: text embeddings | about 0.6 GB | Apache-2.0 |
| Qwen2.5-1.5B-Instruct (4-bit MLX or Q4_K_M GGUF) | optional query parsing | about 0.9 or 1.1 GB | Apache-2.0 |
| InsightFace `buffalo_l` | `people scan`, and `sort`/`find` | a few hundred MB, from GitHub into `~/.insightface/models/` | non-commercial research only |

The image and text halves are one model: the index records the model's name, and searching it with
a different model is refused. The text tower runs on the CPU through onnxruntime without PyTorch:
it loads in about 1 second and takes tens of milliseconds per query. (MobileCLIP, which appears in
some other tools, is not used here.)

Sizes are of the files as downloaded; the numbers above are rounded from the files on disk. The
first `index` run is therefore a download of roughly 1.5 GB plus the text tower on first search,
and the first `people scan` adds InsightFace's pack.

## The index

**Location.** `~/.facesorter/index.sqlite` by default. The environment variable
`FACESORTER_INDEX` overrides it, and `--index PATH` on any command overrides both. The web app also
keeps a cache of 512 px JPEG thumbnails of the photos it has shown in `~/.facesorter/thumbs`,
capped at 2 GiB (least recently used removed first). Both are derived from your photos and contain
personal information: do not share or commit them (the repository's `.gitignore` excludes
`.facesorter/`, `*.sqlite`, `*.sqlite3`, `*.db` and `feedback*.jsonl`).

**Contents.** One SQLite file holds the metadata, the embeddings, face boxes and embeddings, names,
the list of indexed library folders, and ratings. It is safe to copy as one file. An index is tied
to one embedding model and schema version; a mismatch is reported instead of being used. Searching
is exact: all image vectors are loaded into memory and compared with a single matrix product, which
takes single-digit milliseconds at around 100,000 photos.

### Lifecycle and caveats

- **Incremental.** A file is skipped when its size and modification time are unchanged. A re-run
  over an unchanged library only lists and `stat`s the files. `--force` re-embeds everything.
- **Identity is the path.** A renamed or moved file is a new photo: it is embedded again and its
  faces and ratings are lost. The old entry is removed on the next complete `index` run.
- **Pruning is per library folder.** After a complete, unlimited scan, photos under the folder you
  indexed that no longer exist are removed from the index. Other indexed folders are untouched.
  If you rename a whole library folder, the old folder's entries stay in the index as ghosts
  until you remove them with `facesorter index --remove-root OLD_FOLDER` (the old folder does not
  need to exist any more), then index the new one. Moving a folder re-embeds its photos, because
  identity is the path.
- **An unreadable folder is protected.** A subfolder that cannot be listed (permissions, I/O
  error) is logged, and its photos are kept in the index rather than treated as deleted.
- **An empty scan is refused.** If a folder scan finds no photos but the index holds photos from
  that folder (an unmounted drive, a moved folder), `index` stops with exit code 1 and changes
  nothing. `--allow-empty-prune` overrides this.
- **Same folder, different spelling.** The same library reached through another case on a
  case-insensitive filesystem, or through a symlink, reuses the spelling stored in the index.
- **Two indexers at once** leave a consistent index but may over-count `new`/`updated`.
- **Search and indexing together.** `facesorter serve` notices when another process updates the
  index and reloads.

## Privacy

- Photos, embeddings, queries and the index never leave your machine. The only network traffic is
  the one-time model downloads listed above: the SigLIP 2 files and the optional language model
  from the Hugging Face Hub, and InsightFace's `buffalo_l` pack (downloaded from GitHub by the
  InsightFace library on the first `people scan`, `sort` or `find`). After that everything works
  offline.
- The web app binds to `127.0.0.1`. Binding anything else is refused unless `--allow-remote` is
  given, and then a warning says it exposes your library to the network.
- The web app checks the `Host` and `Origin` headers of requests against an allowlist, so a web
  page open in your browser cannot talk to it through DNS rebinding or cross-site requests.
- The web app serves **only its thumbnail cache directory**, never the library: originals are not
  reachable from the browser. Gradio's own cache copies are kept inside that same directory, which
  is created readable by you only (mode 0700). The cache directory must not overlap the library.
  The page shows the file name and the name of the containing folder of a result, and never prints the full library path.
  Thumbnails are small copies of your photos; delete `~/.facesorter/thumbs` at any time.
- Gradio analytics and sharing are switched off (`share` is always false), and Hugging Face Hub
  telemetry is disabled when the app starts. There is no telemetry in facesorter.
- `facesorter eval` prints aggregates only by default (see below).
- Use the tool only on photos you have the right to process. Face recognition on photos of other
  people may need their consent where you live.

## Evaluation

`facesorter eval` measures search quality without captions or manual labels, from two things the
library already has.

1. **Metadata probes.** It samples photos (`--sample`, `--seed`) and builds queries from each
   photo's own date and place ("photos from March 2024", "photos in Paris"); the parsed filters
   must be the expected ones, every hit must satisfy them, and the photo must be among the
   candidates. *Lift probes* check that visual queries land where they should: `snow on the ground`
   should return photos taken disproportionately in winter months, `the eiffel tower` photos
   labelled France, compared with the library's base rate (judged only when the library has enough
   photos with the needed metadata, otherwise "insufficient data"). Add your own with `--queries`.
   A *safety battery* of tricky phrasings (`since 2020`, `03/04/2024`, an unknown place, ...) is
   run in rules mode and must never apply a wrong filter silently.
2. **Your thumbs up/down ratings** from the web app: precision of the rated hits per query and
   overall, reported once there are enough ratings.

It also reports place-label round trips and search latency. What it does not measure is how well
the picture model understands your photos beyond those proxies.

By default the report contains only aggregates: counts, rates and histograms in which any bucket
under 5 photos is folded into "other" and library-wide counts are rounded. File paths, picture
content and the text of rated queries appear only with `--show-paths`. Progress goes to stderr and
the report to stdout, so you can redirect it (`facesorter eval --no-llm > report.md`) or use
`--markdown FILE` and `--json`.

## Limitations

- **Visual search is as good as the model.** On the COCO 5k test set, a public image-text retrieval
  benchmark, the model reaches 53.2% text-to-image Recall@1; the older SigLIP got 47.8%, OpenCLIP
  DataComp B/16 40.2% and the original OpenAI CLIP B/16 30.4%. Photos of your own life are
  not COCO captions: expect wrong hits, especially for abstract, counting, text-in-image and
  fine-grained queries. The ranking is only as good as the top results you can see; there is no
  confidence cut-off.
- **Small hand-labelled checks are optimistic.** The query-reading figures in this project come
  from a few dozen hand-written queries that were also used while developing the prompts and rules.
  They show the approach is sound, not that it will be right on your queries.
- **English queries only**, still photos only. Videos are skipped.
- **One of each filter.** One date range, one place, any number of people, all combined with AND.
- **Date filters need EXIF or folder dates; place filters need GPS.** Photos without them never
  match those filters. WhatsApp-style images and screenshots usually have neither.
- **Names that are not recognised stay in the visual text** in rules mode, and are then matched
  against the picture rather than the metadata.
- **Face grouping is not benchmarked** and chains (see [People](#people)). It is capped at 10,000
  faces.
- **Large libraries are slow to index once.** Decoding 12-megapixel HEIC photos dominates:
  roughly 0.18 s per photo overall in measurements on an Apple-silicon Mac, with the model itself
  at about 30 to 45 ms per photo on the Apple GPU. A few thousand photos take minutes, a hundred
  thousand take hours. Search stays fast either way.
- **No cloud, multi-user or sync**, no approximate index (not needed at this scale), no
  automatic rename-tracking.

### Where the numbers in this guide come from

| Claim | Source | Caveat |
| --- | --- | --- |
| COCO 5k text-to-image R@1: SigLIP 2 B/16 53.2%, SigLIP 47.8%, OpenCLIP DataComp B/16 40.2%, OpenAI CLIP B/16 30.4% | public benchmark | COCO captions, not personal photos; fp16 gave the same R@1 as fp32 |
| Date accuracy of query parsing: 43 to 63% when a 1.5B-class model computes dates, 83 to 100% when it only copies the phrase and code resolves it | small hand-labelled set of queries | optimistic: small, and prompts and rules were tuned on it |
| Fully correct parses with grounding and the default Qwen2.5-1.5B 4-bit: 97% | the same kind of small set | optimistic, as above |
| Exact search in single-digit milliseconds at about 100,000 vectors | timed locally | NumPy matrix product on one machine; scales linearly |
| ONNX text tower loads in about 1 second, tens of milliseconds per query | timed locally | CPU only |
| MLX parsing about 1.3 s per query, 1.4 GB | timed locally | one Apple-silicon Mac |

## Troubleshooting

- **`Photo search needs Python 3.10 or newer` or an optional-package message.** Install with
  `pip install "facesorter[search]"` in a Python 3.10+ environment. Exit code 2.
- **macOS: `ImportError` mentioning `__thread_bss` when importing scipy.** Seen on macOS 27 (beta)
  with Python 3.10 and scipy 1.15.x wheels. Use Python 3.11 or newer, or `pip install "scipy<1.15"`.
  `facesorter --help` still works because heavy imports are deferred.
- **The first run seems stuck.** The first `index` downloads about 1.5 GB, and PyTorch's first
  start-up is slow. Run with `-v` to see the download notice. The first `search` downloads the
  text tower (~0.6 GB). Both are cached afterwards.
- **`No photo index at ...`.** Run `facesorter index LIBRARY` first, or point `--index`/
  `FACESORTER_INDEX` at the right file. Exit code 2.
- **`Unknown model ...` or a model mismatch.** The index was built with a different embedding model
  than the one installed. Re-index into a new file.
- **HEIC photos fail to open.** HEIC support comes from `pillow-heif`, a core dependency; check
  that it installed (`pip show pillow-heif`) and, with `-v`, read the listed reasons. Photos that
  fail are retried on the next run.
- **Date filters match nothing.** Run `facesorter stats`: if most photos are `mtime` or `none` under
  "Date source", they have no usable capture date. The empty-result hint also prints the date span
  the library covers.
- **Place filters match nothing.** Check `With a place` in `facesorter stats`, and that the name is
  spelled the way the gazetteer knows it. The empty-result hint lists the most common places.
- **`--llm mlx` or `--llm llama-cpp` fails.** The extras are described under [Install](#install);
  MLX needs Apple silicon and Python 3.11+. `--llm auto` falls back to rules instead of failing.
- **`serve` refuses to start.** A non-loopback `--host` needs `--allow-remote`; and the index must
  have been built with `facesorter index`.
- **Different results between runs of `search` and `serve`.** The UI defaults to 24 results and
  the CLI to 10; use `-k 24` to compare.

## Exit codes

Same convention as `sort` and `find`.

| Code | Meaning |
| --- | --- |
| `0` | Success, including a search with no matches |
| `1` | The run failed: an unexpected error, an encoder or other runtime failure, an index/model mismatch, an empty scan refused, nothing could be copied by `--copy-to` |
| `2` | Bad arguments (missing query, invalid date, unknown `--place` or `--person`, unknown cluster id, duplicate name), a missing folder or index, a missing optional dependency or model, or a refused `--host` |
| `130` | Interrupted (Ctrl-C) |

## Licences

This project is MIT licensed ([LICENSE](../LICENSE)). The models it downloads have their own terms;
check them before commercial use.

- SigLIP 2 and the Qwen2.5-1.5B-Instruct models used by default are Apache-2.0.
- InsightFace's pretrained `buffalo_l` pack, used for people grouping, is available for
  non-commercial research purposes only. The InsightFace library itself is MIT.
- Place data is from GeoNames, CC BY 4.0, via the MIT-licensed `geonamescache` package.
