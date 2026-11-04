from __future__ import annotations

import logging
import os
from pathlib import Path
from unittest import mock

import pytest
from conftest import NO_FACE

from facesorter import __version__
from facesorter.cli import LOG_NAME, build_parser, main, setup_logging, teardown_logging


@pytest.fixture
def use_fake_model(monkeypatch, fake_model):
    """Make both commands use the fake model instead of loading InsightFace."""
    monkeypatch.setattr("facesorter.pipeline.get_face_model", lambda: fake_model)
    monkeypatch.setattr("facesorter.finder.get_face_model", lambda: fake_model)
    return fake_model


@pytest.fixture
def sort_args(photo_dir):
    """Argument list for a ``sort`` run over the standard photo library."""
    return ["sort", "--source", photo_dir.source, "--output", photo_dir.output]


@pytest.fixture
def find_args(photo_dir, reference_dir):
    """Argument list for a ``find`` run over the standard photo library."""
    return [
        "find",
        "--source",
        photo_dir.source,
        "--reference",
        reference_dir,
        "--output",
        photo_dir.output,
    ]


class TestInformationalFlags:
    def test_version(self, capsys):
        with pytest.raises(SystemExit) as exc:
            main(["--version"])
        assert exc.value.code == 0
        assert capsys.readouterr().out.strip() == f"facesorter {__version__}"

    @pytest.mark.parametrize("argv", [["--help"], ["-h"], ["sort", "--help"], ["find", "--help"]])
    def test_help(self, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 0
        assert "usage: facesorter" in capsys.readouterr().out

    @pytest.mark.parametrize(
        "argv",
        [
            [],
            ["sort"],
            ["sort", "--source", "a"],
            ["find", "--source", "a", "--output", "b"],
            ["frobnicate"],
            ["sort", "--source", "a", "--output", "b", "--threshold", "abc"],
        ],
    )
    def test_usage_errors_exit_2(self, argv, capsys):
        with pytest.raises(SystemExit) as exc:
            main(argv)
        assert exc.value.code == 2
        assert "usage:" in capsys.readouterr().err

    @pytest.mark.parametrize("argv", [["-v", "sort"], ["sort", "-v"]])
    def test_verbose_accepted_before_and_after_subcommand(self, argv):
        base = ["--source", "s", "--output", "o"]
        assert build_parser().parse_args(argv + base).verbose is True
        assert build_parser().parse_args(["sort"] + base).verbose is False


class TestSortCommand:
    def test_end_to_end(self, sort_args, photo_dir, use_fake_model, tree):
        assert main(sort_args) == 0
        files = tree(photo_dir.output)
        assert {"Person_0/alice_1.png", "Person_1/bob_1.PNG", "Person_1/group.png"} <= files
        assert {"report.txt", LOG_NAME} <= files

    def test_report_and_runtime_log_are_separate(self, sort_args, photo_dir, use_fake_model):
        main(sort_args)
        report = Path(os.path.join(photo_dir.output, "report.txt")).read_text(encoding="utf-8")
        log = Path(os.path.join(photo_dir.output, LOG_NAME)).read_text(encoding="utf-8")
        assert "nofaces.png" in report
        assert "Face Sorting Report" not in log
        assert "Starting face sorting pipeline" in log
        assert "Processing complete" in log
        assert "[INFO]" in log

    def test_log_is_overwritten_on_each_run(self, sort_args, photo_dir, use_fake_model):
        main(sort_args)
        main(sort_args)
        log = Path(os.path.join(photo_dir.output, LOG_NAME)).read_text(encoding="utf-8")
        assert log.count("Starting face sorting pipeline") == 1

    def test_output_inside_source_is_allowed(self, photo_dir, use_fake_model, tree):
        out = os.path.join(photo_dir.source, "sorted")
        args = ["sort", "--source", photo_dir.source, "--output", out]
        assert main(args) == 0
        assert main(args) == 0
        assert "Person_0/alice_1.png" in tree(out)

    def test_no_images_exits_1(self, tmp_path, use_fake_model, capsys):
        (tmp_path / "empty").mkdir()
        argv = ["sort", "--source", str(tmp_path / "empty"), "--output", str(tmp_path / "o")]
        assert main(argv) == 1
        assert "No image files found" in capsys.readouterr().err

    def test_no_faces_exits_1(self, tmp_path, make_image, use_fake_model, capsys):
        make_image(tmp_path / "src" / "a.png", NO_FACE)
        argv = ["sort", "--source", str(tmp_path / "src"), "--output", str(tmp_path / "o")]
        assert main(argv) == 1
        assert "No faces" in capsys.readouterr().err
        assert "No faces" in Path(tmp_path / "o" / LOG_NAME).read_text(encoding="utf-8")

    def test_copy_failure_exits_1(self, sort_args, use_fake_model, monkeypatch, capsys):
        def boom(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr("facesorter.pipeline.copy_image_to_person_folder", boom)
        assert main(sort_args) == 1
        assert "Could not copy any files" in capsys.readouterr().err


class TestFindCommand:
    def test_end_to_end(self, find_args, photo_dir, use_fake_model, tree):
        assert main(find_args) == 0
        files = tree(photo_dir.output)
        assert {"alice_1.png", "alice_1_1.png", "alice_2.png", "group.png"} <= files
        assert LOG_NAME in files
        assert not any(name.startswith("bob") for name in files)

    def test_reference_without_faces_exits_1(self, photo_dir, tmp_path, make_image, use_fake_model):
        make_image(tmp_path / "ref" / "r.png", NO_FACE)
        argv = [
            *["find", "--source", photo_dir.source, "--reference", str(tmp_path / "ref")],
            *["--output", photo_dir.output],
        ]
        assert main(argv) == 1

    def test_empty_source_exits_1(self, tmp_path, reference_dir, use_fake_model):
        (tmp_path / "empty").mkdir()
        argv = [
            *["find", "--source", str(tmp_path / "empty"), "--reference", reference_dir],
            *["--output", str(tmp_path / "o")],
        ]
        assert main(argv) == 1

    def test_copy_failure_exits_1(self, find_args, use_fake_model, monkeypatch):
        def boom(*args, **kwargs):
            raise PermissionError("read-only")

        monkeypatch.setattr("facesorter.finder.copy_image", boom)
        assert main(find_args) == 1

    def test_threshold_flag_changes_matches(self, find_args, photo_dir, use_fake_model, tree):
        assert main([*find_args, "--threshold", "1.5"]) == 0
        assert "bob_1.PNG" in tree(photo_dir.output)


class TestPathValidation:
    def test_missing_source_exits_2(self, tmp_path, capsys):
        argv = ["sort", "--source", str(tmp_path / "nope"), "--output", str(tmp_path / "o")]
        assert main(argv) == 2
        assert "source directory does not exist" in capsys.readouterr().err
        assert not (tmp_path / "o").exists()

    def test_missing_reference_exits_2(self, photo_dir, tmp_path, capsys):
        argv = [
            *["find", "--source", photo_dir.source, "--reference", str(tmp_path / "nope")],
            *["--output", photo_dir.output],
        ]
        assert main(argv) == 2
        assert "reference directory does not exist" in capsys.readouterr().err

    def test_output_equal_to_source_exits_2_and_creates_nothing(self, photo_dir, capsys):
        before = sorted(os.listdir(photo_dir.source))
        assert main(["sort", "--source", photo_dir.source, "--output", photo_dir.source]) == 2
        assert "must not be the same as, or contain" in capsys.readouterr().err
        assert sorted(os.listdir(photo_dir.source)) == before
        assert LOG_NAME not in before

    def test_source_inside_output_exits_2_and_creates_nothing(self, photo_dir):
        parent = os.path.dirname(photo_dir.source)
        before = sorted(os.listdir(parent)), sorted(os.listdir(photo_dir.source))
        assert main(["sort", "--source", photo_dir.source, "--output", parent]) == 2
        assert (sorted(os.listdir(parent)), sorted(os.listdir(photo_dir.source))) == before
        assert not os.path.exists(os.path.join(parent, LOG_NAME))

    def test_reference_inside_output_exits_2(self, photo_dir, reference_dir):
        argv = [
            *["find", "--source", photo_dir.source, "--reference", reference_dir],
            *["--output", os.path.dirname(reference_dir)],
        ]
        assert main(argv) == 2

    def test_output_that_is_a_file_exits_2(self, photo_dir, tmp_path, capsys):
        out = tmp_path / "out.txt"
        out.write_text("x")
        assert main(["sort", "--source", photo_dir.source, "--output", str(out)]) == 2
        assert "not a directory" in capsys.readouterr().err

    def test_uncreatable_output_exits_2(self, photo_dir, tmp_path, capsys):
        blocker = tmp_path / "blocker"
        blocker.write_text("x")
        argv = ["sort", "--source", photo_dir.source, "--output", str(blocker / "sub")]
        assert main(argv) == 2
        assert "cannot use output directory" in capsys.readouterr().err


class TestConfig:
    def test_explicit_missing_config_exits_2(self, sort_args, capsys):
        assert main([*sort_args, "--config", "nope.ini"]) == 2
        assert "Config file not found" in capsys.readouterr().err

    def test_implicit_missing_config_is_fine(self, sort_args, use_fake_model, isolated_cwd):
        assert not (isolated_cwd / "config.ini").exists()
        assert main(sort_args) == 0

    def test_invalid_config_exits_2(self, sort_args, write_config, capsys):
        write_config("[DEFAULT]\nconfidence = 7\n")
        assert main(sort_args) == 2
        assert "confidence must be between" in capsys.readouterr().err

    def test_unparsable_config_exits_2(self, sort_args, write_config):
        write_config("this is not ini")
        assert main(sort_args) == 2

    def test_invalid_cli_value_exits_2(self, sort_args, capsys):
        assert main([*sort_args, "--threshold", "5"]) == 2
        assert "command-line options" in capsys.readouterr().err

    def test_implicit_config_is_used(self, sort_args, write_config, use_fake_model):
        write_config("[DEFAULT]\nconfidence = 0.999\n")  # above every fake face score
        assert main(sort_args) == 1

    def test_cli_flag_beats_config_file(self, sort_args, write_config, use_fake_model):
        write_config("[DEFAULT]\nconfidence = 0.999\n")
        assert main([*sort_args, "--confidence", "0.5"]) == 0

    def test_explicit_config_path(self, sort_args, write_config, use_fake_model):
        path = write_config("[DEFAULT]\nconfidence = 0.999\n", name="custom.ini")
        assert main([*sort_args, "--config", path]) == 1

    def test_validation_happens_before_anything_is_created(self, sort_args, photo_dir):
        main([*sort_args, "--confidence", "9"])
        assert not os.path.exists(photo_dir.output)


class TestSettingsReachTheCommands:
    """Patch the heavy entry points to see exactly what the CLI passes down."""

    def test_sort_threshold_maps_to_cluster_threshold(self, sort_args, write_config):
        write_config(
            "[DEFAULT]\ncluster_threshold = 0.4\nmatch_threshold = 0.3\nconfidence = 0.7\n"
        )
        with mock.patch("facesorter.pipeline.process_images") as run:
            assert main(sort_args) == 0
            assert run.call_args.kwargs["cluster_threshold"] == 0.4
            assert run.call_args.kwargs["confidence"] == 0.7
            assert main([*sort_args, "--threshold", "0.2", "--confidence", "0.9"]) == 0
            assert run.call_args.kwargs["cluster_threshold"] == 0.2
            assert run.call_args.kwargs["confidence"] == 0.9

    def test_find_threshold_maps_to_match_threshold(self, find_args, write_config):
        write_config("[DEFAULT]\ncluster_threshold = 0.4\n")
        with mock.patch("facesorter.finder.find_matches") as run:
            assert main(find_args) == 0
            assert run.call_args.kwargs["threshold"] == 0.4  # falls back to cluster_threshold
            assert main([*find_args, "--threshold", "0.25"]) == 0
            assert run.call_args.kwargs["threshold"] == 0.25
            assert (
                run.call_args.kwargs["reference_dir"]
                == find_args[find_args.index("--reference") + 1]
            )


class TestErrorHandling:
    @pytest.mark.parametrize(
        ("error", "code"),
        [
            ("NoImagesError", 1),
            ("NoFacesError", 1),
            ("CopyError", 1),
            ("InvalidPathError", 2),
            (KeyboardInterrupt, 130),
            (RuntimeError, 1),
        ],
    )
    def test_exit_codes(self, sort_args, error, code):
        from facesorter import errors

        exc = getattr(errors, error)("boom") if isinstance(error, str) else error("boom")
        with mock.patch("facesorter.pipeline.process_images", side_effect=exc):
            assert main(sort_args) == code

    def test_expected_errors_print_one_line_message(self, sort_args, capsys):
        from facesorter.errors import NoFacesError

        with mock.patch("facesorter.pipeline.process_images", side_effect=NoFacesError("nobody")):
            main(sort_args)
        err = capsys.readouterr().err
        assert "nobody" in err
        assert "Traceback" not in err

    def test_unexpected_error_is_one_liner_without_verbose(self, sort_args, capsys):
        with mock.patch("facesorter.pipeline.process_images", side_effect=RuntimeError("kaboom")):
            assert main(sort_args) == 1
        err = capsys.readouterr().err
        assert "Unexpected RuntimeError: kaboom" in err
        assert "Traceback" not in err

    def test_unexpected_error_shows_traceback_with_verbose(self, sort_args, capsys):
        with mock.patch("facesorter.pipeline.process_images", side_effect=RuntimeError("kaboom")):
            assert main(["-v", *sort_args]) == 1
        assert "Traceback" in capsys.readouterr().err

    def test_keyboard_interrupt_message(self, sort_args, capsys):
        with mock.patch("facesorter.pipeline.process_images", side_effect=KeyboardInterrupt):
            assert main(sort_args) == 130
        assert "Interrupted" in capsys.readouterr().err

    def test_error_is_also_written_to_runtime_log(self, sort_args, photo_dir):
        with mock.patch("facesorter.pipeline.process_images", side_effect=RuntimeError("kaboom")):
            main(sort_args)
        log = Path(os.path.join(photo_dir.output, LOG_NAME)).read_text(encoding="utf-8")
        assert "kaboom" in log


class TestLogging:
    """The ``facesorter`` logger must be left exactly as a library user would expect."""

    @staticmethod
    def assert_pristine():
        logger = logging.getLogger("facesorter")
        assert logger.level == logging.NOTSET
        assert logger.propagate is True
        assert [type(h) for h in logger.handlers] == [logging.NullHandler]

    def test_handlers_removed_after_success(self, sort_args, use_fake_model):
        assert main(sort_args) == 0
        self.assert_pristine()

    @pytest.mark.parametrize("error", [RuntimeError("x"), KeyboardInterrupt()])
    def test_handlers_removed_after_failure(self, sort_args, error):
        with mock.patch("facesorter.pipeline.process_images", side_effect=error):
            main(sort_args)
        self.assert_pristine()

    def test_handlers_removed_after_path_error(self, tmp_path):
        main(["sort", "--source", str(tmp_path / "nope"), "--output", str(tmp_path / "o")])
        self.assert_pristine()

    def test_handlers_removed_after_config_error(self, sort_args):
        main([*sort_args, "--confidence", "9"])
        self.assert_pristine()

    def test_setup_then_teardown_restores_logger(self, tmp_path):
        handlers = setup_logging(str(tmp_path / "log.txt"), verbose=True)
        logger = logging.getLogger("facesorter")
        assert logger.level == logging.DEBUG
        assert logger.propagate is False
        assert len(handlers) == 2
        teardown_logging(handlers)
        self.assert_pristine()

    def test_teardown_closes_the_log_file(self, tmp_path):
        handlers = setup_logging(str(tmp_path / "log.txt"))
        file_handler = handlers[0]
        assert file_handler.stream is not None
        teardown_logging(handlers)
        assert file_handler.stream is None

    def test_repeated_calls_do_not_duplicate_log_lines(self, sort_args, photo_dir, use_fake_model):
        for _ in range(3):
            main(sort_args)
        log = Path(photo_dir.output, LOG_NAME).read_text(encoding="utf-8")
        assert log.count("Processing complete") == 1

    @pytest.mark.parametrize(("flags", "level"), [([], logging.INFO), (["-v"], logging.DEBUG)])
    def test_verbose_sets_log_level_during_the_run(self, sort_args, flags, level):
        seen = []
        with mock.patch(
            "facesorter.pipeline.process_images",
            side_effect=lambda **kw: seen.append(logging.getLogger("facesorter").level),
        ):
            assert main([*flags, *sort_args]) == 0
        assert seen == [level]
