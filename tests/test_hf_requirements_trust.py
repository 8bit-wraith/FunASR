"""Exercise the actual HF loader without importing ML dependencies or installing packages."""
import ast
import os
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch


class RequirementsTrustTest(unittest.TestCase):
    def setUp(self):
        source = Path(__file__).resolve().parents[1] / "funasr/download/download_model_from_hub.py"
        tree = ast.parse(source.read_text())
        function = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "download_from_hf")
        namespace = {"os": os, "name_maps_hf": {}, "DictConfig": type("DictConfig", (), {})}
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
        self.load = namespace["download_from_hf"]
        self.install = Mock()
        module = types.ModuleType("funasr.utils.install_model_requirements")
        module.install_requirements = self.install
        self.modules = patch.dict(sys.modules, {module.__name__: module})
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.model = Path(self.directory.name)
        self.requirements = self.model / "requirements.txt"
        self.requirements.write_text("synthetic-package-never-install==0\n")

    def test_default_does_not_install(self):
        self.load(model=str(self.model))
        self.install.assert_not_called()

    def test_explicit_false_does_not_install(self):
        self.load(model=str(self.model), trust_remote_code=False)
        self.install.assert_not_called()

    def test_explicit_true_installs_requirements(self):
        result = self.load(model=str(self.model), trust_remote_code=True)
        self.install.assert_called_once_with(str(self.requirements))
        self.assertEqual(result["model_path"], str(self.model))

    def test_model_config_cannot_grant_trust(self):
        (self.model / "config.yaml").write_text("synthetic config")
        (self.model / "model.pt").touch()
        config = {"model": "synthetic", "trust_remote_code": True}
        omega = types.SimpleNamespace(load=lambda _: config, merge=lambda a, b: {**a, **b})
        self.load.__globals__["OmegaConf"] = omega
        self.load(model=str(self.model))
        self.install.assert_not_called()

    def test_string_true_is_not_explicit_boolean_trust(self):
        self.load(model=str(self.model), trust_remote_code="true")
        self.install.assert_not_called()

    def test_trusted_model_without_requirements_does_not_install(self):
        self.requirements.unlink()
        self.load(model=str(self.model), trust_remote_code=True)
        self.install.assert_not_called()


if __name__ == "__main__":
    unittest.main()
