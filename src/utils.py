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

def get_clairvoyant_truthful(budget, my_valuation, m_t, n_auctions):
    # the clairvoyant knows the max bid at each round
    ## I compute my sequence of utilities at every round
    utility = (my_valuation-m_t)*(my_valuation>=m_t)
    # recall that operations with ndarray produce ndarray

    ## Now I have to find the sequence of m_t summing up to budget B and having the maximum sum of utility
    ## In second price auctions, I can find the sequence **greedily**:
    sorted_round_utility = np.flip(np.argsort(utility)) # sorted rounds, from most profitable to less profitable
    clairvoyant_utilities = np.zeros(n_auctions)
    clairvoyant_bids= np.zeros(n_auctions)
    clairvoyant_payments = np.zeros(n_auctions)
    c = 0 # total money spent
    i = 0 # index over the auctions
    while c <= budget-1 and i < n_auctions:
        clairvoyant_bids[sorted_round_utility[i]] = 1 # bid 1 in the remaining most profitable auction
        # recall that since this is a second-price auction what I pay doesn't depend on my bid (but determines if I win)
        # notice that since the competitors' bids are fixed < 1 the clairvoyant can just bid 1 to the auctions he wants to win and 0 to the rest
        clairvoyant_utilities[sorted_round_utility[i]] = utility[sorted_round_utility[i]]
        clairvoyant_payments[sorted_round_utility[i]] = m_t[sorted_round_utility[i]]
        c += m_t[sorted_round_utility[i]]
        i+=1
    return clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments

def plot_clayrvoyant(budget, clairvoyant_bids, clairvoyant_utilities, clairvoyant_payments):
    plt.title('Clairvoyant Bids')
    plt.plot(clairvoyant_bids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Chosen Bids')
    plt.show()

    plt.title('Clairvoyant Cumulative Payment')
    plt.plot(np.cumsum(clairvoyant_payments))
    plt.axhline(budget, color='red', linestyle='--', label='Budget')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum m_t~ 1_{b_t > m_t}$')
    plt.show()

    plt.title('Clairvoyant Cumulative Utility')
    plt.plot(np.cumsum(clairvoyant_utilities))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum u_t$')
    plt.show()

def plot_agent(budget, agent_bids, agent_utilities, agent_payments):
    plt.title('Agent Bids')
    plt.plot(agent_bids)
    plt.xlabel('$t$')
    plt.ylabel('$b_t$')
    plt.title('Chosen Bids')
    plt.show()

    plt.title('Agent Cumulative Payment')
    plt.plot(np.cumsum(agent_payments))
    plt.axhline(budget, color='red', linestyle='--', label='Budget')
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum m_t~ 1_{b_t > m_t}$')
    plt.show()

    plt.title('Agent Cumulative Utility')
    plt.plot(np.cumsum(agent_utilities))
    plt.legend()
    plt.xlabel('$t$')
    plt.ylabel('$\sum u_t$')
    plt.show()

def plot_regret(agent_utilities, clairvoyant_utilities):
    regret = np.cumsum(clairvoyant_utilities) - np.cumsum(agent_utilities)
    plt.title('Agent Regret')
    plt.plot(regret)
    plt.xlabel('$t$')
    plt.ylabel('Regret')
    plt.show()

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