class Instance(object):
    def __init__(self, tokens, slu_tags=None, domain=None, template_list: list=None):
        self.tokens = tokens
        self.slu_tags = slu_tags
        self.template_list = template_list
        self.domain = domain

    def __str__(self):
        return str(self.__dict__)

    def __repr__(self):
        return str(self.__dict__)
