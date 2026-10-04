# Contributing

Thanks for helping keep this list accurate. This repository is a *bibliographic* resource, so
the bar for a new entry is deliberately high.

## The one hard rule: every entry needs a resolvable identifier

An entry without a working DOI, arXiv identifier or publisher URL **will not be merged**.
Do not add a guessed link. If you cannot find the record, do not add the paper.

## Adding a paper

1. **Open an issue first** if you are adding more than a few papers, so we can agree on scope.
2. For each paper, verify the metadata against a primary source, in this order of preference:
   - the publisher's own page or DOI landing page,
   - DBLP (`https://dblp.org/search/publ/api?q=...&format=json`),
   - arXiv (`https://arxiv.org/abs/<id>`),
   - Crossref (`https://api.crossref.org/works/<doi>`).
3. Copy the title **verbatim** from that record. Do not normalise casing, fix what you believe
   are typos, or shorten a truncated title — several entries in the earlier version of this list
   had to be rolled back precisely because of this.
4. Choose the section by task, not by the paper's marketing:
   - **FTSE** — Extrapolation: extends a series beyond the observed window.
   - **FTSI** — Imputation: fills missing values inside the observed window.
   - **FTSS** — Synthesis: generates new series, order flow or market scenarios.
   If a paper uses a generative model on financial data that is *not* a time series (tabular
   credit risk, text augmentation, anomaly detection), put it under *Adjacent & Non-Time-Series
   Work* and say why in a note.
5. Write the abstract in English, in your own words, based on the paper. Do not machine-copy a
   publisher abstract without checking it, and do not invent numbers.

## Entry format

```markdown
#### `Nickname` — Full title as registered

- **Authors**: A. Author, B. Author, C. Author
- **Venue**: Full venue name, Year
- **Link**: <https://doi.org/10.xxxx/yyyyy>
- **Code**: <https://github.com/...>
- **Task**: Short noun phrase
- **Abstract**: Two to four sentences.
```

If a record is genuinely ambiguous, keep the entry and add a blockquote immediately after it:

```markdown
> **Needs verification**: explain exactly what is unresolved
```

Silently dropping an entry is worse than flagging it.

## Before you open a PR

- [ ] Every new entry has a working link — click each one.
- [ ] Titles match the registered record character-for-character.
- [ ] Author lists are complete or truncated with a deliberate `et al.`
- [ ] No duplicate entry (check by DOI, not by title — two papers can share a method nickname).
- [ ] The `Nickname` is unique within the file.
- [ ] The paper has not already been withdrawn or retracted.

## Reporting a metadata error

Open an issue with the entry's nickname, the field that is wrong, and a link to the correct
record. Metadata corrections are always welcome and will be credited.
