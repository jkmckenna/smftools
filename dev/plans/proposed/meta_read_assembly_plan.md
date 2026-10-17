# Meta-read assembly from fragmented amplicon libraries (`MRA`)

Status: proposed. Items `MRA-01`--`MRA-09`. Builds on span-agnostic duplicate
detection (`DSA`, `in-progress/duplicate_detection_span_agnostic_implementation_plan.md`)
and on processing lineages (descendant generations).

## Why

Rapid (transposase) barcoding fragments each PCR amplicon copy, so a read is
a random piece of one copy. For conversion chemistries -- deamination, EM /
bisulfite conversion -- a modification is a sequence change in the original
molecule, and PCR copies it: every amplicon copy of one original molecule
carries the same pattern. Overlapping fragments of different copies of the
same molecule therefore agree where they overlap and can be laid end to end
into a **meta read**: the original molecule, longer than any of its fragments,
with every position supported by one or more reads. Gains: more full-span
molecules and more sites per molecule from the same sequencing, and a
consensus that corrects per-read call errors.

Duplicate detection already finds these groups: `DSA` banding compares reads
of different spans over their overlap, and union-find joins them into
`duplicate_cluster_id` clusters. Today a cluster is reduced to one keeper
(`_select_duplicate_keeper`) and the other members are dropped.

Not for direct-modification calls: PCR erases native marks, and those
libraries are not PCR amplicons.

## Risks the design must answer

- **False merges.** The duplicate distance is the mismatch fraction over at
  least `duplicate_detection_min_overlapping_positions` (20) jointly observed
  sites, threshold 0.07; unmodified sites count like modified ones. Two nearly
  unmodified reads of different molecules agree everywhere and are joined --
  the low-deamination over-collapse seen on real data. For dedup that loses a
  read; for assembly it fabricates a chimeric molecule.
- **Transitive chains.** Union-find clusters are transitive: A~B and B~C join
  A and C even when they do not overlap or would disagree. A cluster is a
  candidate, not an assembly.
- **PCR chimeras** (template switching) carry two molecules' patterns in one
  read and would bridge clusters.

## Design

1. **Evidence per overlap** (stricter than dedup): an edge between two
   fragments needs overlap >= `meta_read_min_overlap` sites, mismatch
   fraction <= `meta_read_max_mismatch`, and at least
   `meta_read_min_shared_modified` sites modified in both -- agreement on
   modified sites is what identifies a molecule; agreement on unmodified
   sites is weak evidence at low deamination.
2. **Consistent layout**: within a candidate cluster, the strict edges form a
   graph; an assembly is a connected set of fragments in which every pair
   that overlaps agrees (re-checked, not inferred through chains) and whose
   reference intervals form one contiguous layout. Clusters with conflicts are
   split into maximal consistent parts (or left as fragments); ambiguity is
   never resolved by guessing.
3. **Consensus**: per position, the majority call over covering fragments,
   with support (fragments covering it) and discordance (any disagreement)
   recorded; positions below `meta_read_min_support` or discordant are masked,
   not called.
4. **Membership rules**: same barcode / sample, same reference strand (v1),
   no chimera-flagged fragment, all fragments passing read QC.

## Data model and downstream use

A meta read is a molecule that is not one sequencing read. It is published
as a **processing lineage** -- the machinery re-basecalled lineages use
(`in-progress/selective_pod5_rebasecalling_implementation_plan.md`): a
descendant preprocess generation whose molecules are the assembled meta
reads plus every fragment that was not assembled, followed by spatial, HMM
and latent descendants, each stage reading the generations its own lineage
published (pinned) rather than whatever the parent selects. The parent chain
stays current and untouched; the lineage appears only when its chain is
complete. In a project, `register_experiment_lineage` records it (e.g.
`meta_reads`) without changing what the project resolves, and
`set_active_lineage` switches an experiment between read-level and meta-read
results -- a selector change, nothing rewritten. One sequencing run thus
yields both versions of every downstream result. Per meta read: a deterministic
`molecule_uid` from its members, member read ids, span, number of fragments,
span gained over the longest member; layers: consensus calls, support,
discordance.

## Items

| id | item | status |
|---|---|---|
| `MRA-01` | strict pairwise evidence (overlap, mismatch, shared modified sites) over `DSA` candidate pairs within duplicate clusters | proposed |
| `MRA-02` | consistent layouts: split clusters into conflict-free contiguous assemblies; chimera-flagged / cross-barcode / cross-strand fragments excluded | proposed |
| `MRA-03` | consensus calls with support and discordance; masking rules | proposed |
| `MRA-04` | publication as the preprocess generation of a `meta_reads` processing lineage (meta reads + unassembled fragments), provenance (member map), molecule identity | proposed |
| `MRA-05` | QC: assembled fraction, span gain, support distribution; a false-merge estimate from impossible merges (fragments of different barcodes that pass the evidence test) | proposed |
| `MRA-06` | config: `preprocess_meta_reads` (off by default), thresholds; deaminase / conversion modalities, locus (amplicon) references | proposed |
| `MRA-07` | validation: simulated fragmentation of full-length reads (with call errors, across deamination levels incl. near-unmodified), and real full-length ligation reads fragmented in silico -- recovered-molecule accuracy and false-merge rate; acceptance before it can be enabled | proposed |
| `MRA-08` | extension: merge complementary-strand fragments of the same molecule (a top-strand C->T and the bottom-strand G->A of the same events) | proposed |
| `MRA-09` | downstream: run spatial / HMM / latent on the lineage with the lineage runner, entered at preprocess (raw shared with the parent; today's runner starts at raw); register it in projects; HMM layer sets per lineage (`HPS-07` reads the current spatial stage today) | proposed |

## Open questions

- Thresholds for shared modified sites per chemistry and dose (the false-merge
  estimate of `MRA-05` should set them).
- Analyses pooling experiments may mix one experiment's meta-read lineage
  with another's read-level chain; molecule identities keep them distinct,
  but such analyses should record which lineage each experiment contributed.
- Whether assembled molecules need a reweighting downstream (a meta read
  stands for one molecule, like a deduplicated read, so counts are unchanged;
  per-position support differs).
