import numpy as np
import matplotlib.pyplot as plt
import numpy as np
""" from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle """

def set_seed(seed):
    np.random.seed(seed)
    return

def generate_adv_sequence(len, min, max):
    return_array = np.zeros(len)
    for i in range(len):
        return_array[i] = np.random.uniform(min, max)
    return return_array

def normalize_zero_one(x, min_x, max_x):
    return (x - min_x) / (max_x - min_x)

def denormalize_zero_one(x, min_x, max_x):
    return min_x + (max_x - min_x) * x


#REPORT AND PLOTS
""" class PDFReport:
    def __init__(self, filename, requirement = 1):
        self.requirement = requirement
        assert self.requirement in [1, 2, 3, 4], "Requirement must be 1 or 2"
        self.doc = SimpleDocTemplate(filename, pagesize=A4)
        self.elements = []
        self.width, self.height = A4
        self.styles = getSampleStyleSheet()
        self.story = []

    def header(self, title = None, params=None):
        # Add Title
        if title is None:
            title = f"Requirement {self.requirement} {params['run_type']}"
        title_style = self.styles['Title']
        self.story.append(Paragraph(title, title_style))
        self.story.append(Spacer(1, 0.25 * inch))

        # Add Parameters
        if params:
            param_data = []
            keys = list(params.keys())
            for i in range(0, len(keys), 3):
                row = [
                    f"{keys[i]}: {params[keys[i]]}",
                    f"{keys[i+1]}: {params[keys[i+1]]}" if i + 1 < len(keys) else "",
                    f"{keys[i+2]}: {params[keys[i+2]]}" if i + 2 < len(keys) else ""
                ]
                param_data.append(row)

            param_table = Table(param_data, colWidths=[self.width/3.5, self.width/3.5, self.width/3.5])
            param_table.setStyle(TableStyle([
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 12),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
            ]))
            self.story.append(param_table)
            self.story.append(Spacer(1, 0.25 * inch))

    def add_text(self, text):
        text_style = self.styles['BodyText']
        self.story.append(Paragraph(text, text_style))
        self.story.append(Spacer(1, 0.25 * inch))

    def add_image(self, image_path, width=4*inch, height=3*inch):
        img = Image(image_path, width=width, height=height)
        self.story.append(img)
        self.story.append(Spacer(1, 0.25 * inch))

    def add_double_image(self, image_path1, image_path2, width=3.5*inch, height=2.5*inch):
        img1 = Image(image_path1, width=width, height=height)
        img2 = Image(image_path2, width=width, height=height)
        
        table_data = [[img1, img2]]
        image_table = Table(table_data, colWidths=[width, width])
        self.story.append(image_table)
        self.story.append(Spacer(1, 0.25 * inch))

    def save(self):
        self.doc.build(self.story) """