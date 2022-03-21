import sys
from text_attack.translation_based.tranlator_attack import BaiDuTranslatorAttacker

PLACE_HOLDER = "PLACE_HOLDER"


class Paraphrase(object):
    def __init__(self, text):
        self.text = text
        self.children = {}
        self.ascedants_cnt = 0

    def __str__(self):
        return self.text


class BFSParaphraser():
    def __init__(self, max_depth):
        self.paraphraser = BaiDuTranslatorAttacker(None)
        self.pivotes = self.paraphraser.get_all_support_lang()
        self.max_translates = len(self.pivotes) ** max_depth
        self.special_sent = PLACE_HOLDER  # just for sprt analysis.

    def extract_texts(self, para_node, sprt=True):
        nodes = [para_node]
        descendants = []
        while len(nodes) > 0:
            node = nodes.pop(0)
            if node.ascedants_cnt < 2:
                for pivot in self.pivotes:
                    if pivot in node.children:
                        descendants.append(str(node.children[pivot]))
                        nodes.append(node.children[pivot])
                    elif sprt:
                        descendants.append(
                            self.special_sent)  # this account method is only designed for this exp and max_depth should be 2.
        return descendants

    def __add_children(self, translates, p_node, asc_number=1):
        for key in translates:
            pivots = translates[key]
            for lang in pivots:
                child = Paraphrase(key)
                child.ascedants_cnt = asc_number
                p_node.children[lang] = child

    def __bfs_search(self, text):
        '''
        search if a descendant node of root node has the same text. if yes, then we return a copy of the node
        or return none. Thus, we can avoid translating a text repeatedly.
        '''

        def is_equal(text1, text2):
            if text1.strip().rstrip(".").lower() == text2.strip().rstrip(".").lower():
                return True
            return False

        queue = [self.root_node]
        while len(queue) > 0:
            node = queue.pop(0)
            if is_equal(node.text, text) and len(node.children) != 0:  # filter the node itself.
                new_node = Paraphrase(node.text)
                for pivot in self.pivotes:
                    if pivot in node.children:
                        new_node.children[pivot] = Paraphrase(node.children[pivot].text)  # not all its descendants
                return new_node
            else:
                for pivot in self.pivotes:
                    if pivot in node.children:
                        queue.append(node.children[pivot])
        return None

    def bfs_parapharase(self, root_node):
        self.root_node = root_node
        self.translates_cnt = 0
        queue = [self.root_node]
        while len(queue) > 0:
            node = queue.pop(0)
            repeated_node = self.__bfs_search(node.text)
            if repeated_node is None:
                translates = self.paraphraser.paraphrase_text(node.text, self.pivotes)
                self.__add_children(translates, node)
            else:
                node = repeated_node
            yield self.__extract_texts(node)
            # add children by the order of pivotes
            # if not is_repeated:
            for pivot in self.pivotes:
                if pivot in node.children:
                    queue.append(node.children[pivot])
            # control the max depth
            self.translates_cnt += len(self.pivotes)
            sys.stdout.write('\r progress:{}/{}'.format(self.translates_cnt, self.max_translates))
            if self.translates_cnt > self.max_translates:
                return self.root_node

    def get_root_node(self):
        return self.root_node
