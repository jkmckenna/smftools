# concatenate_fastqs_to_bam

def concatenate_fastqs_to_bam(
    fastq_files,
    output_bam,
    barcode_tag='BC',
    gzip_suffixes=('.gz',),
    barcode_map=None,
    add_read_group=True,
    rg_sample_field=None,
    progress=True,
    auto_pair=True,
):
    """
    Concatenate FASTQ(s) into an unaligned BAM. Supports single-end and paired-end (auto-detect or explicit).

    Parameters
    ----------
    fastq_files : list[str] or list[(str,str)]
        If list of tuples: each tuple is (R1_path, R2_path).
        If list of strings and auto_pair=True: the function will attempt to automatically pair files.
    output_bam : str
        Path to output BAM (will be overwritten).
    barcode_tag : str
        SAM tag used for barcode (default 'BC').
    gzip_suffixes : tuple
        Compressed suffixes to consider (default ('.gz',)).
    barcode_map : dict or None
        Optional mapping {path: barcode} to override automatic extraction.
    add_read_group : bool
        If True, add RG entries and set RG tag per-read (ID = barcode).
    rg_sample_field : str or None
        If set, includes SM field in RG header entries.
    progress : bool
        Show tqdm progress bar.
    auto_pair : bool
        If True and `fastq_files` is a list of strings, attempt to auto-pair R1/R2 by filename patterns.

    Returns
    -------
    dict
        Summary: {'total_reads', 'per_file_counts', 'paired_count', 'unpaired_count', 'barcodes'}
    """
    import os
    import re
    import gzip
    from itertools import zip_longest
    from Bio import SeqIO
    import pysam
    from tqdm import tqdm

    # ---------- helpers ----------
    def _is_gz(path):
        pl = path.lower()
        return any(pl.endswith(suf) for suf in gzip_suffixes)

    def _strip_fastq_ext(basn):
        # remove .fastq.gz .fq.gz .fastq .fq
        for ext in ('.fastq.gz', '.fq.gz', '.fastq', '.fq'):
            if basn.lower().endswith(ext):
                return basn[:-len(ext)]
        # fallback remove last suffix
        return os.path.splitext(basn)[0]

    def _extract_barcode_from_filename(path):
        # heuristic: barcode is last underscore-separated token in filename (before ext)
        stem = _strip_fastq_ext(os.path.basename(path))
        if '_' in stem:
            token = stem.split('_')[-1]
            if token:
                return token
        # fallback to whole stem
        return stem

    # pairing heuristics: try to identify suffix that marks read number
    def _classify_read_token(stem):
        # returns (prefix, readnum) if matches, else (None, None)
        patterns = [
            r'(?i)(.*?)[._-]r?([12])$',  # prefix_R1 or prefix.r1 or prefix-1
            r'(?i)(.*?)[._-]read[_-]?([12])$',
            r'(?i)(.*?)[/_]([12])$',      # sometimes /1 is used (rare in filenames)
        ]
        for pat in patterns:
            m = re.match(pat, stem)
            if m:
                prefix = m.group(1)
                num = m.group(2)
                return prefix, int(num)
        return None, None

    def pair_by_filename(paths):
        # paths: list of strings
        map_pref = {}  # prefix -> {1: path, 2: path, 'orphans': [..]}
        unpaired = []
        for p in paths:
            name = os.path.basename(p)
            stem = _strip_fastq_ext(name)
            pref, num = _classify_read_token(stem)
            if pref is not None:
                entry = map_pref.setdefault(pref, {})
                entry[num] = p
            else:
                # try fallback: split by last underscore or dot and check last token is 1/2 or R1/R2
                toks = re.split(r'[_\.]', stem)
                if toks and toks[-1] in ('1', '2', 'R1', 'R2', 'r1', 'r2'):
                    last = toks[-1]
                    basepref = "_".join(toks[:-1]) if len(toks) > 1 else toks[0]
                    num = 1 if last.endswith('1') else 2
                    entry = map_pref.setdefault(basepref, {})
                    entry[num] = p
                else:
                    unpaired.append(p)
        pairs = []
        leftovers = []
        for k, d in map_pref.items():
            if 1 in d and 2 in d:
                pairs.append((d[1], d[2]))
            else:
                # put whoever exists into leftovers
                leftovers.extend([v for kk, v in d.items()])
        # append also unpaired
        leftovers.extend(unpaired)
        return pairs, leftovers

    # ---------- normalize input ----------
    explicit_pairs = []
    singles = []
    if not isinstance(fastq_files, (list, tuple)):
        raise ValueError("fastq_files must be a list of paths or list of (R1,R2) tuples.")

    # mixture: if user supplied tuples -> treat as explicit pairs
    if all(isinstance(x, (list, tuple)) and len(x) == 2 for x in fastq_files):
        explicit_pairs = [(str(a), str(b)) for a, b in fastq_files]
    else:
        # flatten and coerce to strings, ignore None
        paths = [str(x) for x in fastq_files if x is not None]
        if auto_pair:
            explicit_pairs, leftovers = pair_by_filename(paths)
            singles = leftovers
        else:
            singles = paths

    # Build barcode map and ordered barcodes
    barcode_map = barcode_map or {}
    per_path_barcode = {}
    barcodes_in_order = []

    # pairs: assign barcode per pair from either provided barcode_map for first file or from filenames
    for r1, r2 in explicit_pairs:
        bc = barcode_map.get(r1) or barcode_map.get(r2) or _extract_barcode_from_filename(r1)
        per_path_barcode[r1] = bc
        per_path_barcode[r2] = bc
        if bc not in barcodes_in_order:
            barcodes_in_order.append(bc)
    for p in singles:
        bc = barcode_map.get(p) or _extract_barcode_from_filename(p)
        per_path_barcode[p] = bc
        if bc not in barcodes_in_order:
            barcodes_in_order.append(bc)

    # prepare BAM header
    header = {"HD": {"VN": "1.0"}, "SQ": []}
    if add_read_group:
        rg_list = []
        for bc in barcodes_in_order:
            rg = {"ID": bc}
            if rg_sample_field:
                rg["SM"] = rg_sample_field
            rg_list.append(rg)
        header["RG"] = rg_list

    # ---------- write BAM ----------
    per_file_counts = {}
    total_written = 0
    paired_count = 0
    unpaired_count = 0

    def _open_fh(path):
        return gzip.open(path, "rt") if _is_gz(path) else open(path, "rt")

    with pysam.AlignmentFile(output_bam, "wb", header=header) as bam_out:
        # process paired files first
        seq_iter = list(explicit_pairs)  # list of (r1,r2)
        if progress:
            seq_iter = tqdm(seq_iter, desc="Paired FASTQ->BAM")
        for r1_path, r2_path in seq_iter:
            if not (os.path.exists(r1_path) and os.path.exists(r2_path)):
                raise FileNotFoundError(f"Paired file missing: {r1_path} or {r2_path}")
            bc = per_path_barcode.get(r1_path) or per_path_barcode.get(r2_path) or "barcode"
            # open both and iterate in parallel
            with _open_fh(r1_path) as fh1, _open_fh(r2_path) as fh2:
                it1 = SeqIO.parse(fh1, "fastq")
                it2 = SeqIO.parse(fh2, "fastq")
                # iterate in lockstep; if one shorter we still write remaining as unpaired (zip_longest)
                for rec1, rec2 in zip_longest(it1, it2, fillvalue=None):
                    # determine a common read name
                    if rec1 is not None:
                        id1 = rec1.id
                    else:
                        id1 = None
                    if rec2 is not None:
                        id2 = rec2.id
                    else:
                        id2 = None
                    # try to derive a common name (strip /1 or /2 if present)
                    def _strip_end_id(s):
                        if s is None:
                            return None
                        return re.sub(r'(?:/1$|/2$|\s[12]$)', '', s)
                    common_name = _strip_end_id(id1) or _strip_end_id(id2) or (id1 or id2)

                    # create AlignedSegment for read1
                    if rec1 is not None:
                        a1 = pysam.AlignedSegment()
                        a1.query_name = common_name
                        a1.query_sequence = str(rec1.seq)
                        a1.is_paired = True
                        a1.is_read1 = True
                        a1.is_read2 = False
                        a1.is_unmapped = True
                        a1.mate_is_unmapped = True
                        # reference fields for unmapped
                        a1.reference_id = -1
                        a1.reference_start = -1
                        a1.next_reference_id = -1
                        a1.next_reference_start = -1
                        a1.template_length = 0
                        # qualities
                        if "phred_quality" in rec1.letter_annotations:
                            try:
                                a1.query_qualities = [int(x) for x in rec1.letter_annotations["phred_quality"]]
                            except Exception:
                                a1.query_qualities = None
                        # tags
                        a1.set_tag(barcode_tag, str(bc), value_type='Z')
                        if add_read_group:
                            a1.set_tag("RG", str(bc), value_type='Z')
                        bam_out.write(a1)
                        per_file_counts.setdefault(r1_path, 0)
                        per_file_counts[r1_path] += 1
                        total_written += 1
                    # create AlignedSegment for read2
                    if rec2 is not None:
                        a2 = pysam.AlignedSegment()
                        a2.query_name = common_name
                        a2.query_sequence = str(rec2.seq)
                        a2.is_paired = True
                        a2.is_read1 = False
                        a2.is_read2 = True
                        a2.is_unmapped = True
                        a2.mate_is_unmapped = True
                        a2.reference_id = -1
                        a2.reference_start = -1
                        a2.next_reference_id = -1
                        a2.next_reference_start = -1
                        a2.template_length = 0
                        if "phred_quality" in rec2.letter_annotations:
                            try:
                                a2.query_qualities = [int(x) for x in rec2.letter_annotations["phred_quality"]]
                            except Exception:
                                a2.query_qualities = None
                        a2.set_tag(barcode_tag, str(bc), value_type='Z')
                        if add_read_group:
                            a2.set_tag("RG", str(bc), value_type='Z')
                        bam_out.write(a2)
                        per_file_counts.setdefault(r2_path, 0)
                        per_file_counts[r2_path] += 1
                        total_written += 1
                    # count paired/unpaired bookkeeping
                    if rec1 is not None and rec2 is not None:
                        paired_count += 1
                    else:
                        # one side missing -> counted as unpaired for whichever exists
                        if rec1 is not None:
                            unpaired_count += 1
                        if rec2 is not None:
                            unpaired_count += 1

        # process singletons
        single_iter = list(singles)
        if progress:
            single_iter = tqdm(single_iter, desc="Single FASTQ->BAM")
        for p in single_iter:
            if not os.path.exists(p):
                raise FileNotFoundError(p)
            bc = per_path_barcode.get(p, "barcode")
            with _open_fh(p) as fh:
                for rec in SeqIO.parse(fh, "fastq"):
                    a = pysam.AlignedSegment()
                    a.query_name = rec.id
                    a.query_sequence = str(rec.seq)
                    a.is_paired = False
                    a.is_read1 = False
                    a.is_read2 = False
                    a.is_unmapped = True
                    a.mate_is_unmapped = True
                    a.reference_id = -1
                    a.reference_start = -1
                    a.next_reference_id = -1
                    a.next_reference_start = -1
                    a.template_length = 0
                    if "phred_quality" in rec.letter_annotations:
                        try:
                            a.query_qualities = [int(x) for x in rec.letter_annotations["phred_quality"]]
                        except Exception:
                            a.query_qualities = None
                    a.set_tag(barcode_tag, str(bc), value_type='Z')
                    if add_read_group:
                        a.set_tag("RG", str(bc), value_type='Z')
                    bam_out.write(a)
                    per_file_counts.setdefault(p, 0)
                    per_file_counts[p] += 1
                    total_written += 1
                    unpaired_count += 1

    summary = {
        "total_reads": total_written,
        "per_file": per_file_counts,
        "paired_pairs_written": paired_count,
        "singletons_written": unpaired_count,
        "barcodes": barcodes_in_order
    }
    return summary


# def concatenate_fastqs_to_bam(fastq_files, output_bam, barcode_tag='BC', gzip_suffix='.gz'):
#     """
#     Concatenate multiple demultiplexed FASTQ (.fastq or .fq) files into an unaligned BAM and add the FASTQ barcode suffix to the BC tag.

#     Parameters:
#         fastq_files (list): List of paths to demultiplexed FASTQ files.
#         output_bam (str): Path to the output BAM file.
#         barcode_tag (str): The SAM tag for storing the barcode (default: 'BC').
#         gzip_suffix (str): Suffix to use for input gzip files (Defaul: '.gz')

#     Returns:
#         None
#     """
#     import os
#     import pysam
#     import gzip
#     from Bio import SeqIO
#     from tqdm import tqdm

#     n_fastqs = len(fastq_files)

#     with pysam.AlignmentFile(output_bam, "wb", header={"HD": {"VN": "1.0"}, "SQ": []}) as bam_out:
#         for fastq_file in tqdm(fastq_files, desc="Processing FASTQ files"):
#             # Extract barcode from the FASTQ filename (handles .fq, .fastq, .fq.gz, and .fastq.gz extensions)
#             base_name = os.path.basename(fastq_file)
#             if n_fastqs > 1:
#                 if base_name.endswith('.fastq.gz'):
#                     barcode = base_name.split('_')[-1].replace(f'.fastq{gzip_suffix}', '')
#                 elif base_name.endswith('.fq.gz'):
#                     barcode = base_name.split('_')[-1].replace(f'.fq{gzip_suffix}', '')
#                 elif base_name.endswith('.fastq'):
#                     barcode = base_name.split('_')[-1].replace('.fastq', '')
#                 elif base_name.endswith('.fq'):
#                     barcode = base_name.split('_')[-1].replace('.fq', '')
#                 else:
#                     raise ValueError(f"Unexpected file extension for {fastq_file}. Only .fq, .fastq, .fq{gzip_suffix}, and .fastq{gzip_suffix} are supported.")
#             else:
#                 barcode = 'barcode0'

#             # Read the FASTQ file (handle gzipped and non-gzipped files)
#             open_func = gzip.open if fastq_file.endswith(gzip_suffix) else open
#             with open_func(fastq_file, 'rt') as fq_in:
#                 for record in SeqIO.parse(fq_in, 'fastq'):
#                     # Create an unaligned BAM entry for each FASTQ record
#                     aln = pysam.AlignedSegment()
#                     aln.query_name = record.id
#                     aln.query_sequence = str(record.seq)
#                     aln.flag = 4  # Unmapped
#                     aln.query_qualities = pysam.qualitystring_to_array(record.letter_annotations["phred_quality"])
#                     # Add the barcode to the BC tag
#                     aln.set_tag(barcode_tag, barcode)
#                     # Write to BAM file
#                     bam_out.write(aln)
