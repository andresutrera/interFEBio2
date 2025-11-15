import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

from interFEBio.Optimize.feb_bindings import (
    BuildContext,
    EvaluationBinding,
    FebBuilder,
    FebTemplate,
    ParameterBinding,
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
  <Control>
    <max_ups>10</max_ups>
  </Control>
</febio_spec>
"""
    p = tmp_path / "base.feb"
    p.write_text(text, encoding="ISO-8859-1")
    return p


def test_template_render_applies_parameter_bindings(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(
        template_path,
        bindings=[
            ParameterBinding(
                theta_name="node_1",
                xpath=".//Geometry/Nodes/node[@id='1']",
            ),
            ParameterBinding(
                theta_name="young",
                xpath=".//Material/material[@id='1']/E",
            ),
            ParameterBinding(
                theta_name="max_ups",
                xpath=".//Control/max_ups",
            ),
        ],
    )

    theta = {"node_1": 2.5, "young": 12.0, "max_ups": 3.0}
    ctx = BuildContext(fmt="%.4f")

    tree = template.render(theta, ctx)
    root = tree.getroot()

    assert root.find(".//Geometry/Nodes/node[@id='1']").text == "2.5000"
    assert root.find(".//Material/material[@id='1']/E").text == "12.0000"
    assert root.find(".//Control/max_ups").text == "3.0000"


def test_parameter_binding_optional_missing_nodes(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(
        template_path,
        bindings=[ParameterBinding(theta_name="unused", xpath=".//missing", required=False)],
    )
    tree = template.render(theta={"unused": 1.0}, ctx=BuildContext())
    assert tree.getroot().tag == "febio_spec"


def test_parameter_binding_missing_value_raises(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(template_path, bindings=[ParameterBinding("foo", ".//Control/max_ups")])
    with pytest.raises(KeyError):
        template.render(theta={}, ctx=BuildContext())


def test_template_write_creates_parent(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(template_path, bindings=[])
    output_dir = tmp_path / "nested" / "caseA"
    out_path = output_dir / "out.feb"

    template.write(theta={}, out_path=str(out_path))

    assert out_path.exists()
    tree = ET.parse(out_path)
    assert tree.getroot().tag == "febio_spec"


def test_evaluation_binding_expression(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(
        template_path,
        bindings=[
            EvaluationBinding(
                xpath=".//Material/material[@id='1']/v",
                value="(node_1 + young) / 2",
            )
        ],
    )

    theta = {"node_1": 4.0, "young": 2.0}
    tree = template.render(theta, ctx=BuildContext(fmt="%.2f"))
    assert tree.getroot().find(".//Material/material[@id='1']/v").text == "3.00"


def test_evaluation_binding_callable(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(
        template_path,
        bindings=[
            EvaluationBinding(
                xpath=".//Control/max_ups",
                value=lambda theta: theta["node_1"] * 10.0,
            )
        ],
    )

    theta = {"node_1": 3.0}
    tree = template.render(theta, ctx=BuildContext(fmt="%.1f"))
    assert tree.getroot().find(".//Control/max_ups").text == "30.0"


def test_builder_creates_case_folder(tmp_path: Path) -> None:
    template_path = _write_template(tmp_path)
    template = FebTemplate(template_path, bindings=[])
    builder = FebBuilder(template=template, subfolder="caseA")

    feb_path = builder.build(
        theta={},
        out_root=str(tmp_path),
        out_name="sim.feb",
        ctx=BuildContext(fmt="%.3f"),
    )

    assert Path(feb_path).exists()
    assert Path(feb_path) == Path(tmp_path) / "caseA" / "sim.feb"
