import pytest
import xml.etree.ElementTree as ET
from src.svg_converter.converter import SVGConverter  

SAMPLE_SVG_INPUT = """
<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100">
    <rect x="10" y="10" width="30" height="30" style="fill:red;stroke:black;stroke-width:2" />
    <circle cx="60" cy="60" r="20" style="fill:blue;stroke:black;stroke-width:1" />
    <line x1="10" y1="90" x2="90" y2="10" style="stroke:black;stroke-width:1" />
</svg>
"""

# Expected output for centralized styles
EXPECTED_STYLE_TAG = """
<style type="text/css">
.st0{fill:red;stroke:black;stroke-width:2;}
.st1{fill:blue;stroke:black;stroke-width:1;}
.st2{stroke:black;stroke-width:1;}
</style>
"""

@pytest.fixture
def load_sample():
    """Fixture to provide sample SVG input."""
    def _load_sample():
        return SAMPLE_SVG_INPUT
    return _load_sample

@pytest.fixture
def save_output(tmp_path):
    """Fixture to save the converted output if needed."""
    def _save_output(content, filename):
        output_file = tmp_path / filename
        output_file.write_text(content)
        return output_file
    return _save_output

def test_svg_conversion(load_sample, save_output):
    """Test the centralization of styles in an SVG."""
    input_svg = load_sample()
    converter = SVGConverter(input_svg)
    converter.centralize_styles()
    output_svg = ET.tostring(converter.root, encoding="unicode")
    save_output(output_svg, "output.svg")

    # Parse the resulting SVG
    output_tree = ET.ElementTree(ET.fromstring(output_svg))
    root = output_tree.getroot()

    defs = root.find(".//{http://www.w3.org/2000/svg}defs")
    assert defs is not None, "Missing <defs> tag in the output SVG"

    style = defs.find(".//{http://www.w3.org/2000/svg}style")
    assert style is not None, "Missing <style> tag in the output SVG"

    # Parse and normalize the style content
    actual_classes = {
        line.split("{")[0].strip("."): line.split("{")[1].strip("}").strip()
        for line in style.text.strip().split("\n")
        if "{" in line and "}" in line
    }
    expected_classes = {
        "st0": "fill:red;stroke:black;stroke-width:2;",
        "st1": "fill:blue;stroke:black;stroke-width:1;",
    }

    # Normalize and compare styles
    for cls, expected_style in expected_classes.items():
        assert cls in actual_classes, f"Missing class: {cls}"
        actual_style = actual_classes[cls]
        assert actual_style.strip(";") == expected_style.strip(";"), (
            f"Mismatch for class {cls}: expected '{expected_style}', got '{actual_style}'"
        )

    # Ensure there are no unexpected classes
    unexpected_classes = set(actual_classes) - set(expected_classes)
    assert not unexpected_classes, f"Unexpected classes: {unexpected_classes}"
