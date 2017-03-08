"""BaseLocalInterpretation class"""
from ..model_interpreter import ModelInterpreter
from lime.lime_tabular import LimeTabularExplainer as l_tab_exp
from lime.lime_text import LimeTextExplainer as l_text_exp
from lime.lime_image import LimeImageExplainer as l_image_exp

class BaseLocalInterpretation(ModelInterpreter):
    """Base class for all local interpretation objects"""
    LimeTabularExplainer = l_tab_exp
    LimeImageExplainer = l_image_exp
    LimeTextExplainer = l_text_exp
