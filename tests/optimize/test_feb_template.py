import xml.etree.ElementTree as ET
from pathlib import Path

from interFEBio.Optimize.feb_bindings import (
    AttrXPathBinding,
    BuildContext,
    CallbackBinding,
    FebBuilder,
    FebTemplate,
    MaterialParamBinding,
    ScalarXPathBinding,
)


def _write_template(tmp_path: Path) -> Path:
    text = """<?xml version="1.0" encoding="ISO-8859-1"?>
<febio_spec>
  <Geometry>
    <Nodes>
      <node id="1">0.0</node>
      <node id="2">0.0</node>
    </Nodes>
  </Geometry>
  <Material>
    <material id="1" name="tissue">
      <E>1.0</E>
      <v>0.30</v>
    </material>
  </Material>
  <Control solver="febiolc" />
</febio_spec>
"""
    p = tmp_path / "base.feb"
    p.write_text(text, encoding="ISO-8859-1")
    return p


def test_template_render_applies_bindings(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(
        template_path,
        bindings=[
            ScalarXPathBinding(
                theta_name="node_1",
                xpath=".//Geometry/Nodes/node[@id='1']",
            ),
            AttrXPathBinding(
                theta_name="solver",
                xpath=".//Control",
                attr="solver",
            ),
            MaterialParamBinding(
                theta_name="young",
                tag_name="E",
                selector=("id", "1"),
            ),
            CallbackBinding(
                fn=lambda root, theta, ctx: ET.SubElement(
                    root.find(".//Material/material"), "custom"
                ).set("value", ctx.fmt % theta["custom"]),
            ),
        ],
    )

    theta = {"node_1": 2.5, "solver": 42, "young": 12.0, "custom": 3.14}
    ctx = BuildContext(iter_id=3, case_name="specimen", fmt="%.4f")

    tree = template.render(theta, ctx)
    root = tree.getroot()

    assert root.find(".//Geometry/Nodes/node[@id='1']").text == "2.5000"
    assert root.find(".//Control").attrib["solver"] == "42.0000"
    assert root.find(".//Material/material/E").text == "12.0000"
    custom = root.find(".//Material/material/custom")
    assert custom is not None and custom.attrib["value"] == "3.1400"


def test_template_write_creates_parent(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(template_path, bindings=[])
    output_dir = tmp_path / "nested" / "caseA"
    out_path = output_dir / "out.feb"

    template.write(theta={}, out_path=str(out_path), ctx=BuildContext(0, "case"))

    assert out_path.exists()
    tree = ET.parse(out_path)
    assert tree.getroot().tag == "febio_spec"


def test_builder_creates_case_folder(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(template_path, bindings=[])
    builder = FebBuilder(template=template, subfolder="caseA")

    feb_path, xplt_path = builder.build(
        theta={},
        out_root=str(tmp_path),
        out_name="sim.feb",
        ctx=BuildContext(iter_id=1, case_name="caseA", fmt="%.3f"),
    )

    assert Path(feb_path).exists()
    assert Path(xplt_path) == Path(tmp_path) / "caseA" / "sim.xplt"
