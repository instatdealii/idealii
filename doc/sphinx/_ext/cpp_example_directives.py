import re

from sphinx.parsers import RSTParser
from docutils.frontend import OptionParser
from sphinx.util.docutils import SphinxDirective
from docutils.utils import new_document

class CppExamplePlainDirective(SphinxDirective):
    required_arguments = 1
    has_content = False

    def run(self) -> list:
        rst = ""
        rst += f".. code-block:: c++ \n"

        file_ = self.arguments[0]
        commentblock_ = False
        
        with open(file_, "r") as f:
            for line in f:
                if commentblock_:
                    match = re.search("^ *\*/",line)
                    if match:
                        commentblock_ = False
                    continue

                # see if we have a single comment line
                match = re.search("^ *//",line)
                if match: 
                    continue
                    
                # see if we have a comment block
                match = re.search("^ */\*",line)
                if match: 
                    commentblock_ = True
                    continue

                rst += f"\t{line}"
        return self.parse_rst(rst)

    def parse_rst(self, text):
        parser = RSTParser()
        parser.set_application(self.env.app)

        settings = OptionParser(
            defaults=self.env.settings,
            components=(RSTParser,),
            read_config_files=True,
        ).get_default_values()
        document = new_document("<rst-doc>", settings=settings)
        parser.parse(text, document)
        return document.children

class CppExampleDirective(SphinxDirective):
    required_arguments = 1
    has_content = False

    def run(self) -> list:
        rst = ""

        file_ = self.arguments[0]
        commentblock_ = False
        insourceblock_ = False
        noteblock_ = False
        codeblock_ = False
        
        with open(file_, "r") as f:
            for line in f:
                if commentblock_:
                    if re.search("^ *\*/",line):
                        rst += f"\n"
                        commentblock_ = False
                        if noteblock_:
                            rst += f"\n"
                            noteblock_ = False
                    else:
                        line = re.sub('^ *\* *','',line)
                        if noteblock_:
                            rst += f"\t{line}"
                        else:
                            rst += f"{line}"
                    continue
                # see if we have a source file only comment (block starting/ending with // --+)
                if re.search("^ *//\*--+",line):
                    if insourceblock_:
                        insourceblock_ = False
                    else:
                        insourceblock_ = True
                    continue

                if insourceblock_:
                    continue

                # ignore heading comment lines (line of /\n// heading name\n line of //)
                if re.search("^ *///+",line):
                    continue

                if re.search("^ *// ?@<H[23]> ?",line):
                    tmp_ = re.sub('^ *// ?@<','',line)
                    headinglevel = tmp_[1]
                    line = re.sub('^ *// ?@<H[23]> ?','',line)
                    rst += f"\n\n{line}"
                    if headinglevel == '2':
                        rst += f"{'-' * len(line)}\n"
                    else:
                        rst += f"{'^' * len(line)}\n"
                    continue                    

                # see if we have a single comment line
                if re.search("^ *//",line): 
                    if codeblock_:
                        rst+= f"\n"
                    codeblock_ = False
                    line = re.sub('^ *// ?','',line)
                    rst += f"{line}"
                    continue

                # see if we have a Note block
                if re.search("^ */(\*)+ Note",line):
                    noteblock_ = True
                    line = re.sub('^ */(\*)+ ','',line)
                    rst += f".. admonition:: {line}\n"
                    commentblock_ = True
                    codeblock_ = False
                    continue
                    
                # see if we have a comment block
                if re.search("^ */(\*)+",line):
                    commentblock_ = True
                    codeblock_ = False
                    continue

                if not codeblock_:
                    if not re.search("^ *$",line):
                        codeblock_=True
                        rst += f"\n.. code-block:: c++ \n \t:dedent: 0\n\n"

                rst += f"\t{line}"
        return self.parse_rst(rst)

    def parse_rst(self, text):
        parser = RSTParser()
        parser.set_application(self.env.app)

        settings = OptionParser(
            defaults=self.env.settings,
            components=(RSTParser,),
            read_config_files=True,
        ).get_default_values()
        document = new_document("<rst-doc>", settings=settings)
        parser.parse(text, document)
        return document.children


def setup(app: object) -> dict:
    app.add_directive("cpp-example-plain", CppExamplePlainDirective)
    app.add_directive("cpp-example", CppExampleDirective)