import datasets

"""
This unit test suite contains various sanity checks for the OAM-TCD dataset
namely:

- Are licenses valid for all datasets and images?
- Is there any overlap between indices in the different training folds and the test set?

"""

# Global load datasets, probably make this a fixture but...
ds = datasets.load_dataset("restor/tcd")
ds_nc = datasets.load_dataset("restor/tcd-nc")
ds_sa = datasets.load_dataset("restor/tcd-sa")


def test_download_dataset():
    assert len(ds) > 0
    assert len(ds_nc) > 0
    assert len(ds_sa) > 0


def test_no_id_overlap():
    # Grab the OAM IDs and fold indices
    filter_cols = ds.select_columns(["validation_fold", "oam_id"])

    # Also check against indices in the test data
    test_ids = ds["test"]["oam_id"]

    # Assert that there's no overlap between folds
    for idx_a in range(5):
        for idx_b in range(5):
            if idx_a == idx_b:
                continue

            rows_a = filter_cols["train"].filter(
                lambda x: x["validation_fold"] == idx_a
            )
            rows_b = filter_cols["train"].filter(
                lambda x: x["validation_fold"] == idx_b
            )

            oam_a = rows_a["oam_id"]
            oam_b = rows_b["oam_id"]

            # check that there are no intersections
            assert (
                len(set(oam_a) & set(oam_b)) == 0
            ), f"ID overlap between folds {idx_a} and {idx_b}"
            assert (
                len(set(oam_a) & set(test_ids)) == 0
            ), f"ID overlap between folds {idx_a} and the test set"
            assert (
                len(set(oam_b) & set(test_ids)) == 0
            ), f"ID overlap between folds {idx_b} and the test set"


def check_licenses(dataset, expected_license):
    for split in dataset:
        # assert dataset.license == expected_license, "Dataset does not use the expected license"
        licenses = set(dataset[split]["license"])
        assert (
            expected_license in licenses
        ), "Could not find expected license in dataset"
        assert len(licenses) == 1, "More than one license found in dataset"


def test_cc_license():
    check_licenses(ds, "CC-BY 4.0")


def test_nc_license():
    check_licenses(ds_nc, "CC BY-NC 4.0")


def test_sa_license():
    check_licenses(ds_sa, "CC BY-SA 4.0")
