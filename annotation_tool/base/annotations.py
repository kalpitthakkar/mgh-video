from .shapes import Rectangle, Circle

class Annotation(object):
    # Limits of the canvas
    keepWithin = Rectangle()

    selectedPart = None
    size_marker = 3
    initialized = False
    parts = {}
    image = None

    wname = ""
    frame_n = 0
    parts_df = []
    colorDict = {}

    def __init__(self, label):
        self.parts[label] = Circle(label=label)
