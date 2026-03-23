#!/usr/bin/env python3
"""Generate type stubs for the RiemannOpt Rust extension module."""

import inspect
import types
from typing import Any, get_type_hints
import sys
import os
from pathlib import Path

# Add the built module to path
sys.path.insert(0, str(Path(__file__).parent / "target/wheels"))

def generate_class_stub(cls: type, indent: str = "") -> str:
    """Generate stub for a class."""
    lines = []

    # Class definition
    bases = ", ".join(base.__name__ for base in cls.__bases__ if base != object)
    if bases:
        lines.append(f"{indent}class {cls.__name__}({bases}):")
    else:
        lines.append(f"{indent}class {cls.__name__}:")

    # Add docstring if available
    if cls.__doc__:
        doc = cls.__doc__.strip()
        if doc:
            lines.append(f'{indent}    """{doc}"""')

    # Get all members
    members = []
    for name, obj in inspect.getmembers(cls):
        if name.startswith("_") and not name.startswith("__"):
            continue
        if name in ["__class__", "__module__", "__dict__", "__weakref__", "__doc__"]:
            continue
        members.append((name, obj))

    if not members and not cls.__doc__:
        lines.append(f"{indent}    ...")
        return "\n".join(lines)

    # Process members
    for name, obj in members:
        if name == "__init__" or name == "__new__":
            # Constructor
            try:
                sig = inspect.signature(obj)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == "self" or param_name == "cls":
                        params.append(param_name)
                    elif param.annotation != inspect.Parameter.empty:
                        params.append(f"{param_name}: {param.annotation}")
                    elif param.default != inspect.Parameter.empty:
                        params.append(f"{param_name}={param.default}")
                    else:
                        params.append(f"{param_name}: Any")

                lines.append(f"{indent}    def {name}({', '.join(params)}) -> None: ...")
            except (ValueError, TypeError):
                lines.append(f"{indent}    def {name}(self, *args, **kwargs) -> None: ...")

        elif callable(obj) and not isinstance(obj, type):
            # Method
            try:
                sig = inspect.signature(obj)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        params.append("self")
                    elif param.annotation != inspect.Parameter.empty:
                        params.append(f"{param_name}: {param.annotation}")
                    else:
                        params.append(f"{param_name}: Any")

                return_type = "Any"
                if sig.return_annotation != inspect.Signature.empty:
                    return_type = str(sig.return_annotation)

                lines.append(f"{indent}    def {name}({', '.join(params)}) -> {return_type}: ...")
            except (ValueError, TypeError):
                lines.append(f"{indent}    def {name}(self, *args, **kwargs) -> Any: ...")

        elif isinstance(obj, property):
            # Property
            lines.append(f"{indent}    @property")
            lines.append(f"{indent}    def {name}(self) -> Any: ...")

    return "\n".join(lines)

def generate_function_stub(func: callable, name: str, indent: str = "") -> str:
    """Generate stub for a function."""
    try:
        sig = inspect.signature(func)
        params = []
        for param_name, param in sig.parameters.items():
            if param.annotation != inspect.Parameter.empty:
                params.append(f"{param_name}: {param.annotation}")
            elif param.default != inspect.Parameter.empty:
                params.append(f"{param_name}={param.default}")
            else:
                params.append(f"{param_name}: Any")

        return_type = "Any"
        if sig.return_annotation != inspect.Signature.empty:
            return_type = str(sig.return_annotation)

        return f"{indent}def {name}({', '.join(params)}) -> {return_type}: ..."
    except (ValueError, TypeError):
        return f"{indent}def {name}(*args, **kwargs) -> Any: ..."

def generate_module_stub(module_name: str) -> str:
    """Generate stubs for a module."""
    lines = [
        '"""Auto-generated type stubs for RiemannOpt Rust extension."""',
        "",
        "from typing import Any, Optional, Tuple, List, Dict, Union, Callable",
        "import numpy as np",
        "import numpy.typing as npt",
        "",
    ]

    try:
        module = __import__(module_name)

        # Get all public members
        for name in dir(module):
            if name.startswith("_") and name != "__version__":
                continue

            obj = getattr(module, name)

            if isinstance(obj, type):
                # Class
                lines.append(generate_class_stub(obj))
                lines.append("")
            elif callable(obj):
                # Function
                lines.append(generate_function_stub(obj, name))
                lines.append("")
            elif isinstance(obj, str) and name == "__version__":
                lines.append(f"__version__: str")
                lines.append("")
    except Exception as e:
        lines.append(f"# Error generating stubs: {e}")

    return "\n".join(lines)

def main():
    """Main entry point."""
    # Try to import the compiled module
    try:
        import _riemannopt

        # Generate stubs for the main module
        stub_content = generate_module_stub("_riemannopt")

        # Write to file
        output_path = Path(__file__).parent / "python" / "riemannopt" / "_riemannopt.pyi"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(stub_content)

        print(f"Generated stubs at {output_path}")

    except ImportError as e:
        print(f"Could not import _riemannopt module: {e}")
        print("Please run 'maturin build' first")
        sys.exit(1)

if __name__ == "__main__":
    main()
