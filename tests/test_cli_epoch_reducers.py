from openbench._cli.eval_command import normalize_epoch_reducers


def test_pass_hat_expands_into_ladder():
    reducers = normalize_epoch_reducers(["pass_hat_3"])
    assert reducers == ["pass_hat_1", "pass_hat_2", "pass_hat_3"]


def test_duplicates_removed_and_order_preserved():
    reducers = normalize_epoch_reducers(["pass_hat_2", "pass_hat_1"])
    assert reducers == ["pass_hat_1", "pass_hat_2"]


def test_comma_separated_entries_supported():
    reducers = normalize_epoch_reducers(["mean,pass_hat_2", "pass_hat_3"])
    assert reducers == ["mean", "pass_hat_1", "pass_hat_2", "pass_hat_3"]
