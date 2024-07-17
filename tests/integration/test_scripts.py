from pytest_console_scripts import ScriptRunner


def test_predict_script(script_runner: ScriptRunner):
    result = script_runner.run(["tcd-predict", "--help"])
    assert result.returncode == 0


def test_predict_script_inference_semantic(script_runner: ScriptRunner):
    result = script_runner.run(
        "tcd-predict semantic tests/images/5b3622402b6a08001185f8d8_10_00004.tif tests/output".split(
            " "
        )
    )
    assert result.returncode == 0


def test_predict_script_inference_instance(script_runner: ScriptRunner):
    result = script_runner.run(
        "tcd-predict instance tests/images/5b3622402b6a08001185f8d8_10_00004.tif tests/output".split(
            " "
        )
    )
    assert result.returncode == 0


def test_predict_script_inference_zoo(script_runner: ScriptRunner):
    from tcd_pipeline.pipeline import known_models

    models = known_models.keys()

    for model in models:
        result = script_runner.run(
            f"tcd-predict {model} tests/images/5b3622402b6a08001185f8d8_10_00004.tif tests/output".split(
                " "
            )
        )
        assert result.returncode == 0


def test_train_script(script_runner: ScriptRunner):
    result = script_runner.run(["tcd-train", "--help"])
    assert result.returncode == 0
