from pathlib import Path
import re
import unittest

ROOT = Path(__file__).resolve().parents[1]

class ReleaseHygieneTests(unittest.TestCase):
    def test_all_project_versions_are_106(self):
        checks = {
            "CMakeLists.txt": r"project\(DiffSamplerNative VERSION 1\.0\.6 LANGUAGES",
            "cpp/diffsampler.cpp": r'constexpr char VERSION\[\] = "1\.0\.6";',
            "diffsampler.py": r'VERSION = "1\.0\.6"',
            "README.md": r"DiffSampler 1\.0\.6",
        }
        for name, pattern in checks.items():
            self.assertRegex((ROOT / name).read_text(encoding="utf-8"), pattern, name)

    def test_model_is_a_sidecar_not_an_executable_resource(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        cpp = (ROOT / "cpp/diffsampler.cpp").read_text(encoding="utf-8")
        rc = (ROOT / "resources/app.rc.in").read_text(encoding="utf-8")
        self.assertIn('copy "${MODEL_FILE}" "${DIST_DIR}/model/pc_nsf_hifigan_44.1k_hop512_128bin_2025.02.onnx"', cmake)
        self.assertIn('set(NOTICE_FILE "${CMAKE_CURRENT_SOURCE_DIR}/model/NOTICE.txt")', cmake)
        self.assertIn("MODEL_RELATIVE_PATH", cpp)
        self.assertNotIn("IDR_VOCODER", cpp)
        self.assertNotIn("@MODEL_FILE_RC@", rc)

    def test_build_does_not_bypass_execution_policy(self):
        build = (ROOT / "build_windows.bat").read_text(encoding="utf-8")
        self.assertNotIn("ExecutionPolicy Bypass", build)

    def test_manifest_is_embedded_only_once_for_msvc(self):
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        rc = (ROOT / "resources/app.rc.in").read_text(encoding="utf-8")
        self.assertIn("/MANIFEST:NO", cmake)
        self.assertEqual(rc.count("1 24 "), 1)

    def test_windows_header_macros_are_not_redefined(self):
        cpp = (ROOT / "cpp/diffsampler.cpp").read_text(encoding="utf-8")
        cmake = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertNotIn("#define NOMINMAX", cpp)
        self.assertNotIn("#define WIN32_LEAN_AND_MEAN", cpp)
        self.assertIn("NOMINMAX WIN32_LEAN_AND_MEAN", cmake)

    def test_standard_windows_metadata_exists(self):
        manifest = (ROOT / "resources/DiffSampler.manifest").read_text(encoding="utf-8")
        rc = (ROOT / "resources/app.rc.in").read_text(encoding="utf-8")
        self.assertIn('requestedExecutionLevel level="asInvoker"', manifest)
        self.assertIn("FILEVERSION 1,0,6,0", rc)
        self.assertIn('OriginalFilename", "DiffSampler.exe', rc)

if __name__ == "__main__":
    unittest.main()
