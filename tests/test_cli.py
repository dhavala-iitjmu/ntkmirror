from ntkmirror.cli import _training_kwargs, build_parser


def test_training_kwargs_uses_v2_validation_and_retain_names():
    args = build_parser().parse_args(
        [
            "fit",
            "--train",
            "train.jsonl",
            "--out",
            "controller.pt",
            "--steps",
            "3",
            "--eval-every",
            "2",
            "--early-stop-patience",
            "1",
            "--retain",
            "retain.jsonl",
            "--retain-weight",
            "0.25",
            "--kl-to-base",
            "0.5",
            "--quiet",
        ]
    )
    kwargs = _training_kwargs(args)
    assert kwargs["validation_interval"] == 2
    assert kwargs["early_stop_patience"] == 1
    assert kwargs["retain_weight"] == 0.25
    assert kwargs["kl_to_base"] == 0.5
    assert kwargs["select_best_on_validation"] is True
    assert kwargs["verbose"] is False


def test_training_kwargs_keeps_legacy_retain_kl_alias():
    args = build_parser().parse_args(
        [
            "fit",
            "--train",
            "train.jsonl",
            "--out",
            "controller.pt",
            "--retain-kl-weight",
            "0.75",
        ]
    )
    assert _training_kwargs(args)["kl_to_base"] == 0.75
