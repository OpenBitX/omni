#!/usr/bin/env python3
"""Print ONNX input/output names + shapes without doing inference."""
import sys
from pathlib import Path

try:
    import onnx
except ImportError:
    print("pip install onnx")
    sys.exit(1)

for p in sys.argv[1:]:
    path = Path(p)
    if not path.exists():
        print(f"{p}: not found")
        continue
    m = onnx.load(str(path))
    print(f"=== {path.name} ===")
    for inp in m.graph.input:
        dims = [d.dim_value if d.dim_value else d.dim_param or "?" for d in inp.type.tensor_type.shape.dim]
        print(f"  in  {inp.name}: {dims}")
    for out in m.graph.output:
        dims = [d.dim_value if d.dim_value else d.dim_param or "?" for d in out.type.tensor_type.shape.dim]
        print(f"  out {out.name}: {dims}")
    print(f"  opset: {[(i.domain or 'ai.onnx', i.version) for i in m.opset_import]}")
    print(f"  size: {path.stat().st_size / 1024 / 1024:.1f} MB")
    print()
