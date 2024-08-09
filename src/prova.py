import matplotlib.pyplot as plt
import numpy as np
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.utils import ImageReader
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle

class PDFReport:
    def __init__(self, filename):
        self.doc = SimpleDocTemplate(filename, pagesize=A4)
        self.elements = []
        self.width, self.height = A4
        self.styles = getSampleStyleSheet()
        self.story = []

    def header(self, title, params=None):
        # Add Title
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
        self.doc.build(self.story)

    def create_plot(self, plot_func, filename):
        plt.figure(figsize=(5, 3))
        plot_func()
        plt.savefig(filename)
        plt.close()
# Usage Example
if __name__ == "__main__":
    report = PDFReport("example_report.pdf")
    report.header("Annual Sales Report", {
        "Author": "John Doe", "Date": "2024-08-09", 
        "Department": "Sales", "Reviewed By": "Jane Smith", 
        "Version": "1.0", "Status": "Draft"
    })    
    # Adding some text
    report.add_text("This is an overview of the sales performance over the last year.")
    
    # Creating and adding the first image (Sine Wave)
    report.create_plot(lambda: plt.plot(np.linspace(0, 10, 100), np.sin(np.linspace(0, 10, 100)), label='Sine'), "sine_wave.png")
    report.add_image("sine_wave.png")

    report.add_text("The following graphs show the trends in sales for the year.\n\n\n\n Now I want to show you what a viking can really do")
    report.add_text("This is a test")
    report.add_text("This is a test")
    report.add_text("This is a test")

    # Creating and adding the second image (Cosine Wave)
    report.create_plot(lambda: plt.plot(np.linspace(0, 10, 100), np.cos(np.linspace(0, 10, 100)), label='Cosine'), "cosine_wave.png")
    report.add_image("cosine_wave.png")

    # Creating and adding the third image (Exponential Curve)
    report.create_plot(lambda: plt.plot(np.linspace(0, 10, 100), np.exp(np.linspace(0, 10, 100) / 3), label='Exponential'), "exponential_curve.png")
    report.add_image("exponential_curve.png", width = 3*inch, height = 2*inch)

    # Creating and adding the fourth image (Random Scatter) in a double image layout
    report.create_plot(lambda: plt.scatter(np.linspace(0, 10, 100), np.random.random(100), label='Random Scatter'), "random_scatter.png")
    report.add_double_image("sine_wave.png", "random_scatter.png")

    # Save the report
    report.save()