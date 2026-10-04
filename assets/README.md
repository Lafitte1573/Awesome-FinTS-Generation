## Data Assets

The `assets/` directory holds the literature corpus used to build the local database and the
survey's literature review. It contains **metadata, notes and images only** — no paper full text.

### Layout

| Path | Contents |
|---|---|
| `assets/index.csv` | Placement index: section → taxonomy family → leaf → entry, with DOI / arXiv id |
| `assets/markdowns/` | Papers converted from PDF to Markdown by OCR |
| `assets/notes/` | LLM-generated structured labels, one JSON per paper |
| `assets/tech-notes/` | LLM-generated reading notes, intended for quick review |
| `assets/nick_names.csv` | Legacy method-nickname assignments from the original repository |
| `assets/cover.png` / `assets/cover.svg` | Cover figure and its vector source |
| `assets/taxonomy.png` | Figure 1 of the survey, exported from the compiled PDF |
| `assets/Survey_00.png` | Legacy cover, superseded by `cover.png` |

### `assets/index.csv`

Header: `section,family,leaf,nick_name,doi,arxiv`.

`section` is one of `FTSE` / `FTSI` / `FTSS` / `COMPARATOR` / `SURVEY` / `ADJACENT`. `family` and
`leaf` follow the two-level taxonomy of Figure 1 of the survey; for `SURVEY` rows, `family` holds the
comparison group (`S1`–`S5`). Rows with an empty `doi` carry an `arxiv` identifier instead.

This file is the machine-readable projection of the README's Literature Review and Related Surveys
chapters. Where the two disagree, the README wins.

### `assets/notes/` schema

Each file is a single JSON object. All values are plain JSON types — the bracketed annotations
below describe the type, they are not part of the data.

```json
{
  "authors": ["string"],
  "year": 2024,
  "research_directions": ["string"],
  "specific_task": ["string"],
  "summary": "string",
  "techniques": ["string"],
  "method": "string",
  "title": "string"
}
```

| Field | Type | Meaning |
|---|---|---|
| `authors` | `string[]` | Author list |
| `year` | `int` | Year of first publication |
| `research_directions` | `string[]` | Research directions |
| `specific_task` | `string[]` | Sub-task(s) |
| `summary` | `string` | One-line summary of the paper |
| `techniques` | `string[]` | Main techniques used |
| `method` | `string` | Method description |
| `title` | `string` | Paper title |

> **Note** — These fields were generated with LLM assistance and are **not** authoritative. The
> verified bibliographic records live in the root [`README.md`](../README.md) and
> [`fts_dataset.json`](../fts_dataset.json); where the two disagree, those win.
