import xml.etree.cElementTree as ET


class plotVar:
    def __init__(self, name: str):
        self.name = name

    def tree(self):
        tree = ET.Element("var", type=self.name)
        return tree
