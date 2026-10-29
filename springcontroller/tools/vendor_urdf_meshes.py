"""Copy a URDF's mesh files into this package and point the URDF at the copies.

Usage: vendor_urdf_meshes.py URDF DEST_DIR PACKAGE_URI_PREFIX NAME=SRC_ROOT ...

Every <mesh filename="..."> under one of the SRC_ROOTs is copied to
DEST_DIR/NAME/<path relative to SRC_ROOT> and rewritten to
PACKAGE_URI_PREFIX/NAME/<same relative path> (e.g.
package://springcontroller/meshes/gen2_robotiq/kinova/meshes/base.dae).
Exits non-zero if any mesh is missing or lies outside every SRC_ROOT -- a
URDF that loads without its collision meshes silently disables
self-collision checking.
"""

import os
import shutil
import sys
import xml.etree.ElementTree as ET


def main() -> None:
    urdf, dest, prefix = sys.argv[1], sys.argv[2], sys.argv[3].rstrip("/")
    roots = [arg.split("=", 1) for arg in sys.argv[4:]]
    roots = [(name, os.path.realpath(path)) for name, path in roots]

    tree = ET.parse(urdf)
    errors, copied = [], 0
    for mesh in tree.getroot().iter("mesh"):
        src = os.path.realpath(mesh.get("filename", ""))
        match = next(((n, r) for n, r in roots if src.startswith(r + os.sep)), None)
        if match is None:
            errors.append(f"outside every source root: {src}")
            continue
        if not os.path.isfile(src):
            errors.append(f"missing: {src}")
            continue
        name, root = match
        rel = os.path.relpath(src, root)
        target = os.path.join(dest, name, rel)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        if not os.path.isfile(target) or os.path.getmtime(target) < os.path.getmtime(src):
            shutil.copy2(src, target)
            copied += 1
        mesh.set("filename", f"{prefix}/{name}/{rel}")

    if errors:
        sys.exit("Mesh problems:\n  " + "\n  ".join(sorted(set(errors))))
    tree.write(urdf, xml_declaration=True, encoding="utf-8")
    print(f"meshes: {copied} copied/updated under {dest}")


if __name__ == "__main__":
    main()
