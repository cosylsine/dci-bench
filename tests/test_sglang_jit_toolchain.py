import ast
import pathlib
import tempfile
import unittest


SOURCE = (
    pathlib.Path(__file__).resolve().parents[1]
    / "sglang/python/sglang/kernels/jit/utils/compile/toolchain.py"
)


class CudaRuntimeLinkTest(unittest.TestCase):
    def link_flags(self, root, with_device=True):
        tree = ast.parse(SOURCE.read_text())
        function = next(
            node for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "base_link_flags"
        )
        namespace = {
            "List": list,
            "pathlib": pathlib,
            "tvm_ffi_paths": lambda: ((), "/ffi/lib", "tvm_ffi"),
            "is_hip_runtime": lambda: False,
            "cuda_home": lambda: str(root),
        }
        exec(compile(ast.Module(body=[function], type_ignores=[]), str(SOURCE), "exec"), namespace)
        return namespace["base_link_flags"](with_device=with_device)

    def test_wheel_versioned_runtime(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "lib").mkdir()
            (root / "lib/libcudart.so.13").touch()
            self.assertEqual(self.link_flags(root)[-2:], [f"-L{root}/lib", "-l:libcudart.so.13"])

    def test_standard_toolkit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = pathlib.Path(tmp)
            (root / "lib64").mkdir()
            (root / "lib64/libcudart.so").touch()
            self.assertEqual(self.link_flags(root)[-2:], [f"-L{root}/lib64", "-lcudart"])

    def test_cpu_module_does_not_link_cuda(self):
        self.assertEqual(self.link_flags("/missing", False), ["-shared", "-L/ffi/lib", "-ltvm_ffi"])
