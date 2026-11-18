# 01 — Import nomenclatures

The vocabulary to learn comes from medical nomenclatures. Each one is declared in `<workdir>/medstt.yaml`, downloaded from its source, and rewritten in one common JSONL format so that the next step ([select words](02-words.md)) does not depend on the source.

## Configure

`medstt init` writes this section. Each downloaded entry needs a `format` and a `url`; `file` (a glob) is only required when the archive contains several data files. The `local` format takes a `path` instead of a `url`.

```yaml
nomenclatures:
  cim10:
    format: cim10-claml
    url: https://www.atih.sante.fr/sites/default/files/content/documents/cim10fr2024syst_claml_20231215.zip
  orphanet:
    format: orphanet-xml
    url: https://www.orphacode.org/data/packs/Orphanet_Nomenclature_Pack_FR.zip
    file: ORPHAnomenclature_fr_*.xml
```

| Format | Source | What is kept |
| --- | --- | --- |
| `cim10-claml` | [CIM-10-FR à usage PMSI](https://www.atih.sante.fr/cim-10-fr-2024-usage-pmsi) (ATIH, ClaML XML in a zip). See the ATIH terms of use. | The 10 040 leaf diagnostic categories: preferred label, inclusion terms as synonyms, definition when present, chapter and block. Chapters, blocks, and categories with subcategories are skipped. Cross-reference codes nested in labels are removed. |
| `orphanet-xml` | [Orphanet nomenclature pack, French](https://www.orphacode.org/pack-nomenclature/) (zip with several XML files, CC-BY 4.0). | All 11 645 clinical entities: name, synonyms, definition (available for about 60%), disorder type, classification level. |
| `local` | Hand-built `*.txt` lists read from a local `path` (default `local`), one term per line. See [local term lists](#local-term-lists). | Every nonempty line that is not a `#` comment, as a label; the file name is the category. |

To add a source, add an importer under `src/medstt/nomenclatures/importers/`, register its format in `IMPORTERS`, and declare it in the YAML.

## Import

```bash
medstt nomenclature import            # every configured nomenclature
medstt nomenclature import orphanet   # one of them
medstt nomenclature import --force    # download and normalize again
```

The command needs network access, except for `local` entries, which are read in place and leave nothing in `raw/`. Only the data file of each download is kept, as `<workdir>/nomenclature/raw/<name>.xml` (the archive and its other files are discarded), and reused on later runs; the normalized file is `<workdir>/nomenclature/normalized/<name>.jsonl`. Prepare the workspace on a connected machine before copying it to an offline cluster.

```text
<workdir>/nomenclature/
  raw/cim10.xml        the ClaML file from the ATIH zip
  raw/orphanet.xml     ORPHAnomenclature_fr_<year>.xml from the Orphanet pack
  normalized/cim10.jsonl
  normalized/orphanet.jsonl
```

## Normalized format

One JSON object per concept:

```json
{"code": "A00.9", "source": "cim10", "label": "Choléra, sans précision", "synonyms": [],
 "definition": null, "category": "A", "parent": "A00",
 "extra": {"chapter": "A", "block": "A00-A09"}}
```

| Field | Meaning |
| --- | --- |
| `code` | Identifier in the source (CIM-10 code, ORPHAcode, `<file>:<line>` for `local`) |
| `source` | Source name: `cim10`, `orphanet`, or `local` |
| `label` | Preferred label |
| `synonyms` | Other labels, deduplicated |
| `definition` | Definition text, or `null` |
| `category` | Chapter letter (CIM-10), disorder type (Orphanet), or file name without extension (`local`) |
| `parent` | Parent code when the source has a hierarchy, else `null` |
| `extra` | Source-specific fields (block, classification level, status, expert link) |

## Local term lists

The `local/` directory holds hand-built term lists that complement the downloaded nomenclatures. They cover vocabulary that no public nomenclature contains: names of people, places, and departments specific to CHU de Montpellier.

| File | Content | Entries |
| --- | --- | --- |
| [`chu_proper_names.txt`](../local/chu_proper_names.txt) | Proper names linked to the CHU: sites, streets, historical figures | 160 |
| [`chu_physicians.txt`](../local/chu_physicians.txt) | Physicians listed in the CHU's public directory | 1 611 |
| [`chu_specialties.txt`](../local/chu_specialties.txt) | Medical specialties | 45 |
| [`chu_departments.txt`](../local/chu_departments.txt) | Departments (services) | 89 |

All four files were collected on 2026-11-16.

### File format

Plain UTF-8 text, one term per line, no header. A term may contain several words (`Gui de Chauliac`, `Avenue du Doyen Gaston Giraud`). Empty lines and lines starting with `#` are ignored.

### Import the lists

`medstt init` copies the bundled `.txt` files into `<workspace>/local/` and
activates this source in new workspaces:

```yaml
nomenclatures:
  local:
    format: local
    path: local

words:
  sources: [cim10, orphanet, local]
```

The lists are also bundled in the installed wheel, so this works without a Git
checkout and from any working directory. Include `local/` when transferring the
workspace to the cluster. Re-running `init` copies missing bundled files without
replacing existing lists or configuration. For an older workspace, run `init`
again and add the YAML entries above to the existing sections.

You can replace `path` with another directory or a single `.txt` file. Relative
paths are searched in the workspace first, then the current directory.

```bash
medstt nomenclature import local
```

The files are read in place: nothing is downloaded or copied to `nomenclature/raw/`. Each line becomes one concept of `nomenclature/normalized/<name>.jsonl`, in the [normalized format](#normalized-format):

```json
{"code": "chu_proper_names:4", "source": "local", "label": "Occitanie", "synonyms": [],
 "definition": null, "category": "chu_proper_names", "parent": null,
 "extra": {"file": "chu_proper_names.txt", "line": 4}}
```

The code is the file name and the line number, so it changes when lines are inserted above. To have [`words build`](02-words.md) read the list, add its name to `words.sources`.

### `chu_proper_names.txt`

Names a clinician may dictate that are tied to the institution rather than to medicine in general:

- current sites and buildings (`Lapeyronie`, `Arnaud de Villeneuve`, `Centre Antonin Balmès`, `Peyre Plantade`);
- former hospitals (`Hôtel-Dieu Saint-Éloi`, `Hôpital suburbain`, `Asile de Font d'Aurelle`);
- streets, districts, and tram stops around the sites (`Avenue Augustin Fliche`, `Plan des Quatre Seigneurs`);
- historical figures of the hospitals and the medical faculty (`Guillaume Rondelet`, `Émile Jeanbrau`);
- general management and supervisory board (`Anne Ferrer`, `Thomas Le Ludec`);
- member hospitals of the GHT Est-Hérault Sud-Aveyron (`Lodève`, `Millau`);
- a few acronyms (`IFMS`, `IURC`, `IRMB`, `SAMU`).

Some names appear in several forms on purpose (`Lapeyronie`, `Hôpital Lapeyronie`, `François Gigot de Lapeyronie`; `Saint-Éloi` and `Saint-Eloi`).

Sources: the [CHU website](https://www.chu-montpellier.fr/fr/a-propos-du-chu/decouvrir), the [Montpellier city archives](https://archives.montpellier.fr/medecine), French Wikipedia ([CHU](https://fr.wikipedia.org/wiki/Centre_hospitalier_universitaire_de_Montpellier), [medical faculty](https://fr.wikipedia.org/wiki/Facult%C3%A9_de_m%C3%A9decine_de_Montpellier), [Hôpitaux-Facultés district](https://fr.wikipedia.org/wiki/H%C3%B4pitaux-Facult%C3%A9s_(Montpellier))), and the ARS Occitanie list of GHT members. Six entries had no source confirming them as written: the first names in `Jules Euzière` and `Jean-Antoine Giral`, `Maternité Grasset`, `Languedoc`, `Avicenne`, and `SAMU`.

### `chu_physicians.txt`

The names published in the CHU's ["Trouver un médecin" directory](https://www.chu-montpellier.fr/fr/a-propos-du-chu/offre-de-soins/medecins), written `LASTNAME Firstname` and sorted by last name. The site often drops accents in first names (`Cedric`, `Clement`) and writes compound last names without hyphens (`CHAPOUTOT CIURANA Florence`); neither was corrected. The directory does not necessarily list every practitioner (residents, for instance), and it changes over time.

### `chu_specialties.txt` and `chu_departments.txt`

The two option lists of the CHU's [department search form](https://www.chu-montpellier.fr/fr/a-propos-du-chu/offre-de-soins/services). Specialties are in upper case, mostly without accents (`ANESTHESIE - REANIMATION`). Department labels keep the site's spelling and case (`dermatologie et oncologie cutanée`, `Radiologie Diagnostique et Interventionnelle St Eloi`) and may contain commas.

### Known limitations

- Entries carry no definition or synonyms, so the sentence prompt gets no context beyond the label and the file name.
- Missing accents and upper-case labels should be fixed before the entries are used for sentence generation or TTS.
- Acronyms and proper names need a decision on how they are spelled out for the TTS.
- `words build` keeps single rare words, not whole terms; its effect on these lists has not been checked.

### Adding a list

Put a `.txt` file in `local/`, add a row to the table at the top of this section, and describe its source and collection date in its own subsection.

Next: [select the rare words](02-words.md).

Downloads and ZIP extraction use a temporary directory. The cached data file is
replaced only after the operation succeeds, including with `--force`. An
interrupted extraction preserves the previous cache and can be retried.
