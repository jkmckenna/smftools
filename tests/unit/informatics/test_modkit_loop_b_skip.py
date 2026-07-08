from smftools.informatics.modkit_extract_to_adata import (
    _individual_mod_dicts_superseded_by_combined,
)


def test_dual_modification_skips_individual_dict_types() -> None:
    # m6A bottom/top (2, 3) and 5mC bottom/top (5, 6) are superseded by the
    # combined dict types (7, 8) once both modifications are requested.
    assert _individual_mod_dicts_superseded_by_combined(["6mA", "5mC"]) == {2, 3, 5, 6}
    assert _individual_mod_dicts_superseded_by_combined(["5mC", "6mA"]) == {2, 3, 5, 6}


def test_single_modification_does_not_skip_anything() -> None:
    # With only one modification requested there is no combined dict type to
    # supersede the individual one -- it must still be built.
    assert _individual_mod_dicts_superseded_by_combined(["6mA"]) == set()
    assert _individual_mod_dicts_superseded_by_combined(["5mC"]) == set()
    assert _individual_mod_dicts_superseded_by_combined([]) == set()
