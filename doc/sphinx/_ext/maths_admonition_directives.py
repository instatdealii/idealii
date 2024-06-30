
import os 
import warnings

from docutils import nodes
from docutils.parsers.rst.directives.admonitions \
import Admonition as AdmonitionDirective
from sphinx.util.docutils import SphinxDirective

class MathsStatementDirective(AdmonitionDirective,SphinxDirective):
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    extra_classes = [ ]

    @classmethod
    def get_cssname(cls):
        return "maths-statement"

    def run(self):
        name = self.get_cssname()
        self.node_class = nodes.admonition
        if len(self.arguments) == 0:
            self.arguments = [name.title()]

        ret = super().run()
        ret[0].attributes['classes'].append(name)
        target_id = name+'-%d' % self.env.new_serialno(name)
        targetnode = nodes.target('','',ids=[target_id])
        ret[0].target_id = target_id
        ret[0].target_docname = self.env.docname 
        ret.insert(0,targetnode)
        return ret 

class MathsEquationDirective(AdmonitionDirective,SphinxDirective):
    required_arguments = 0
    optional_arguments = 1
    final_argument_whitespace = True
    extra_classes = [ ]

    @classmethod
    def get_cssname(cls):
        return "maths-equation"

    def run(self):
        name = self.get_cssname()
        self.node_class = nodes.admonition
        if len(self.arguments) == 0:
            self.arguments = [name.title()]

        ret = super().run()
        ret[0].attributes['classes'].append(name)
        target_id = name+'-%d' % self.env.new_serialno(name)
        targetnode = nodes.target('','',ids=[target_id])
        ret[0].target_id = target_id
        ret[0].target_docname = self.env.docname 
        ret.insert(0,targetnode)
        return ret 

def init_static_path(app):
    static_path = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                               '_static'))
    app.config.html_static_path.append(static_path)

def setup(app):
    "Maths admonitions setup"
    app.add_directive("maths-statement",MathsStatementDirective)
    app.add_directive("math-statement", MathsStatementDirective)
    app.add_directive("maths-equation",MathsEquationDirective)
    app.add_directive("math-equation", MathsEquationDirective)
    app.connect('builder-inited',init_static_path)
    app.add_css_file("maths_admonition.css")

    return {
        'version': 0.1,
        'parallel_read_safe': True,
        'parallel_write_safe': True,
    }