import xml.etree.ElementTree as ET
import re

class SVGConverter:
    def __init__(self, input_svg):
        self.input_svg = input_svg
        self.tree = ET.ElementTree(ET.fromstring(input_svg))
        self.root = self.tree.getroot()

    def centralize_styles(self):
        """Centralize the styles by extracting, normalizing, and centralizing."""
        # Step 1: Find all unique classes used in the SVG
        used_classes = self._extract_used_classes()

        # Step 2: Extract all unique styles and map them to classes
        style_map = self._extract_styles(used_classes)

        # Step 3: Add centralized styles to <defs> tag inside <style>
        self._add_styles_to_defs(style_map)

        # Step 4: Update elements with the new class-based styles
        self._update_elements_with_class(style_map)

    def _extract_used_classes(self):
        """Extract all used class names from the SVG elements."""
        used_classes = set()
        for elem in self.root.findall(".//*[@class]"):
            classes = elem.attrib["class"].split()
            used_classes.update(classes)
        return used_classes

    def _extract_styles(self, used_classes):
        """Extract all styles associated with the used classes."""
        style_map = {}
        for elem in self.root.findall(".//*[@style]"):
            style = elem.attrib["style"]
            class_name = elem.attrib.get("class")
            if class_name:
                for cls in class_name.split():
                    # Normalize the style before mapping
                    normalized_style = self._normalize_style(style)
                    style_map[cls] = normalized_style  
        return style_map

    def _add_styles_to_defs(self, style_map):
        """Add centralized styles to the <defs> tag inside <style>."""
        defs = self.root.find(".//{http://www.w3.org/2000/svg}defs")
        if defs is None:
            defs = ET.SubElement(self.root, "{http://www.w3.org/2000/svg}defs")

        style_elem = defs.find(".//{http://www.w3.org/2000/svg}style")
        if style_elem is None:
            style_elem = ET.SubElement(defs, "{http://www.w3.org/2000/svg}style")

        # Create style rules for each class
        style_text = "\n".join(
            f".{cls} {{{style}}}" for cls, style in style_map.items()
        )
        style_elem.text = style_text

    def _update_elements_with_class(self, style_map):
        """Update SVG elements to use the class names."""
        for elem in self.root.findall(".//*[@style]"):
            style = elem.attrib["style"]
            class_name = self._get_class_for_style(style, style_map)
            if class_name:
                elem.attrib["class"] = class_name

    def _get_class_for_style(self, style, style_map):
        """Generate or retrieve the class name for a given style."""
        normalized_style = self._normalize_style(style)
        for cls, mapped_style in style_map.items():
            if normalized_style == mapped_style:
                return cls
        new_class = f"st{len(style_map)}"
        style_map[new_class] = normalized_style
        return new_class

    def _normalize_style(self, style):
        """Normalize a style string into a dictionary."""
        style_dict = {}
        for rule in style.strip().split(';'):
            rule = rule.strip()
            if rule:
                key, value = rule.split(':', 1)
                key = key.strip()
                value = value.strip()
                value = self._normalize_value(value)
                style_dict[key] = value
        return style_dict

    def _normalize_value(self, value):
        """Normalize style values (e.g., removing units where possible)."""
        value = re.sub(r'(\d+)(px|em|%)', r'\1', value)  
        return value

    def _style_matches_class(self, style, class_name, style_map):
        """Check if the style matches an existing class's style."""
        normalized_style = self._normalize_style(style)
        class_style = style_map.get(class_name)
        return normalized_style == class_style

    def get_svg(self):
        """Return the modified SVG as a string."""
        return ET.tostring(self.root, encoding="unicode")
