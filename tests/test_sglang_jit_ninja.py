import unittest
from pathlib import Path
from types import SimpleNamespace


REPO_ROOT = Path(__file__).resolve().parents[1]
NINJA_SOURCE = (
    REPO_ROOT
    / "sglang/python/sglang/kernels/jit/utils/compile/ninja.py"
)


class SGLangJitNinjaTest(unittest.TestCase):
    def test_cuda_rule_passes_selected_host_compiler_to_nvcc(self):
        """Keep the launcher-selected CXX compiler effective for CUDA JIT."""
        source = NINJA_SOURCE.read_text(encoding="utf-8")
        for statement in (
            "from sglang.kernels.jit.utils.common import is_hip_runtime, is_musa_runtime\n",
            "from sglang.kernels.jit.utils.compile import toolchain\n",
            "from sglang.kernels.jit.utils.compile.spec import BuildSpec\n",
        ):
            source = source.replace(statement, "")

        class Toolchain:
            @staticmethod
            def compilers():
                return "/opt/gcc-12/bin/g++", "/usr/local/cuda/bin/nvcc"

            @staticmethod
            def base_include_paths():
                return []

            @staticmethod
            def base_cxx_flags():
                return []

            @staticmethod
            def base_cuda_flags():
                return []

            @staticmethod
            def target_flags():
                return []

            @staticmethod
            def base_link_flags(*, with_device):
                if not with_device:
                    raise AssertionError("the probe must exercise CUDA JIT")
                return []

        namespace = {
            "toolchain": Toolchain,
            "is_hip_runtime": lambda: False,
            "is_musa_runtime": lambda: False,
        }
        exec(compile(source, str(NINJA_SOURCE), "exec"), namespace)

        unit = SimpleNamespace(is_cuda=True, stem="kernel", filename="kernel.cu")
        spec = SimpleNamespace(
            translation_units=lambda: [unit],
            include_paths=(),
            cflags=(),
            cuda_cflags=(),
            ldflags=(),
            module_name="sgl_kernel_jit_host_cc_probe",
        )
        generated = namespace["generate"](spec)

        self.assertIn(
            'command = $nvcc -ccbin $cxx -MD -MF "$out.d" $cudaflags -c "$in" -o "$out"',
            generated,
        )


if __name__ == "__main__":
    unittest.main()
