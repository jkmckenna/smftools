## Informatics: `inform`

## Informatics module diagram
```{image} ../_static/smftools_informatics_diagram.png
:width: 1000px
```

Processes raw sequencing data to load an adata object.

```{eval-rst}
.. autosummary::
   :toctree: generated/informatics
   
   smftools.informatics.analysis_region_plan
   smftools.informatics.artifact_paths
   smftools.informatics.bam_functions
   smftools.informatics.basecalling
   smftools.informatics.bed_functions
   smftools.informatics.binarize_converted_base_identities
   smftools.informatics.complement_base_list
   smftools.informatics.converted_BAM_to_adata
   smftools.informatics.derived_read_index
   smftools.informatics.fasta_functions
   smftools.informatics.fastq_export
   smftools.informatics.h5ad_functions
   smftools.informatics.incremental_zarr
   smftools.informatics.modkit_extract_to_adata
   smftools.informatics.modkit_functions
   smftools.informatics.molecule_identity
   smftools.informatics.ohe
   smftools.informatics.partition_read
   smftools.informatics.partition_query
   smftools.informatics.partition_store
   smftools.informatics.physical_layout
   smftools.informatics.plot_region_stitching
   smftools.informatics.pod5_functions
   smftools.informatics.ragged_store
   smftools.informatics.raw_store
   smftools.informatics.reference_identity
   smftools.informatics.region_catalog
   smftools.informatics.run_multiqc
   smftools.informatics.sequence_encoding
   smftools.informatics.sidecar_manifest
   smftools.informatics.signal_features
   smftools.informatics.storage_planner
```

The `ragged_store`/`raw_store`/`partition_read`/`partition_store` modules implement the v2.x
partitioned storage architecture: a read-relative ragged parquet source of truth, a thin
molecule-index `spine.h5ad`, and on-demand dense-slice materialization. `molecule_identity` and
`derived_read_index` provide project-wide molecule keys and searchable raw-to-derived task
lineage. `fastq_export` and `sequence_encoding` build on the ragged store to reconstruct literal
read sequence/quality for FASTQ export.
`partition_query` prunes molecule and derived-task Parquet indexes before opening array stores,
then projects Zarr rows, genomic columns, and requested layers before bounded conversion to memory.
`artifact_paths` keeps cross-stage pointers relative to the run or project root when the platform
supports it, so a complete dataset tree remains readable after copying or renaming.
`region_catalog` validates the three original-coordinate BED scopes and publishes the mapping from
alignment/reduced/conversion records and stored strand references back to the original FASTA.
`analysis_region_plan` consumes that inherited mapping to define shared, non-overlapping analysis
cores for partition-aware stages.
`plot_region_stitching` resolves presentation intervals against completed task cores, selects
molecules deterministically before materialization, and records the exact task and layer
provenance used by each stitched plot.

```{eval-rst}
.. automodule:: smftools.informatics
   :no-members:
   :show-inheritance:
```


### Diagram of final steps of Direct SMF workflow
```{image} ../_static/modkit_extract_to_adata.png
:width: 1000px
```

### Diagram of final steps of Conversion SMF workflow
```{image} ../_static/converted_BAM_to_adata.png
:width: 1000px
```
