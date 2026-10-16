"""Add a small placeholder <inertial> to a URDF link that has none.

Usage: add_placeholder_inertial.py URDF LINK_NAME MASS_KG
"""

import sys
import xml.etree.ElementTree as ET


def main() -> None:
    path, link_name, mass = sys.argv[1], sys.argv[2], sys.argv[3]
    tree = ET.parse(path)
    for link in tree.getroot().iter("link"):
        if link.get("name") == link_name and link.find("inertial") is None:
            inertial = ET.SubElement(link, "inertial")
            ET.SubElement(inertial, "mass", value=mass)
            ET.SubElement(inertial, "origin", xyz="0 0 0", rpy="0 0 0")
            ET.SubElement(inertial, "inertia", ixx="1e-4", ixy="0", ixz="0",
                          iyy="1e-4", iyz="0", izz="1e-4")
    tree.write(path, xml_declaration=True, encoding="utf-8")


if __name__ == "__main__":
    main()
