import argparse
import inspect
import io
import re
import tokenize
from typing import Dict, List, NamedTuple

KEY_INFO_TUPLE = NamedTuple("KEY_INFO_TUPLE", [("help_str", str),
                                               ("choices", List),
                                               ("display", bool)])
KEY_INFO_DICT = Dict[str, KEY_INFO_TUPLE]


class ProgramArgs:
    """Configure the args of a program as fast as possible!

    - attributes = definition of args
    - comments = help texts + choices of args

    Examples:
    >>> # test.py
    >>> class Config(ProgramArgs):
    >>>     def __init__(self):
    >>>         super().__init__()
    >>>         # 1. comments will become a help text in command line
    >>>         #    by running `python test.py --help`
    >>>         # 2. choices of an arg will starts with an "@", running
    >>>         #    `python test.py --model=bert` will throw an error
    >>>         self.model = "cnn"  # neural network @cnn/boe/lstm
    >>>         self.layer = 3 # @2/3/4
    >>>      
    >>>         # display: off
    >>>         # 3. args here can not be printed
    >>>         self.this_will_not_be_printed = 1
    >>>         self.due_to_the_display_flag = 2
    >>>         # display: on
    >>>          
    >>>         self.loss = 'nll'  # loss function
    >>> 
    >>> if __name__ == "__main__":
    >>>     config = Config()._parse_args()
    >>>     print(config)
    If you run `python test.py --model=lstm`, you will get:
        Basic Args:
            --model=lstm
            --layer=3
            --loss=nll
    """
    def __init__(self):
        self._KEY_INFOS_: KEY_INFO_DICT = {}

    def __repr__(self):
        basic_ret = ""
        for key, value in self.__dict__.items():
            if key == "_KEY_INFOS_" or not self._KEY_INFOS_[key].display:
                continue
            basic_ret += "\t--{}={}\n".format(key, value)

        deduced_ret = ""
        for ele in dir(self):
            if ele[0] == '_' or \
                ele in self.__dict__ or \
                    inspect.ismethod(getattr(self, ele)):
                continue
            deduced_ret += "\t--{}={}\n".format(ele, getattr(self, ele))

        ret = "Basic Args:\n" + basic_ret
        if deduced_ret != "":
            ret += "Deduced Args:\n" + deduced_ret
        return ret

    def _parse_args(self):
        self._parse_key_infos()

        parser = argparse.ArgumentParser()
        bool_keys = []
        for key, value in self.__dict__.items():
            if key == "_KEY_INFOS_":
                continue
            # Hack support for true/false
            if isinstance(value, bool):
                bool_keys.append(key)
                value = str(value)

            parser.add_argument('--{}'.format(key),
                                action='store',
                                default=value,
                                type=type(value),
                                help=self._KEY_INFOS_[key].help_str,
                                choices=self._KEY_INFOS_[key].choices,
                                dest=str(key))
        parsed_args = parser.parse_args().__dict__
        for ele in bool_keys:
            if parsed_args[ele] in ['True', 'true', 'on', '1', 'yes']:
                parsed_args[ele] = True
            elif parsed_args[ele] in ['False', 'false', 'off', '0', 'no']:
                parsed_args[ele] = False
            else:
                raise Exception(
                    'You must pass a boolean value for arg {}'.format(ele))
        self.__dict__.update(parsed_args)
        # self._check_args()
        return self

    def _parse_key_infos(self):
        defs, _ = inspect.getsourcelines(self.__init__)
        display = True
        for line in defs:
            line = line.strip()
            # Single line comment
            if line.startswith("#"):
                line = line.replace(" ", "")
                if line == '#display:off':
                    display = False
                if line == '#display:on':
                    display = True
                continue
            # Comment after code
            found = re.search(r"self\s*.([^\s]+)\s*=.*", line)
            if found:
                key = found.group(1)
                comment = get_comment(line)
                help_str, choices = None, None
                if comment:
                    if "@" in comment:
                        help_str, choices = comment.split("@")
                        help_str = help_str.strip()
                        choices = choices.split("/")
                        choices = list(map(lambda x: x.strip(), choices))
                    else:
                        help_str = comment
                self._KEY_INFOS_[key] = KEY_INFO_TUPLE(help_str=help_str,
                                                       choices=choices,
                                                       display=display)

    # def _check_args(self):
    #     pass
    #     for key, value in self.__dict__.items():
    #         if key == "_KEY_INFOS_":
    #             continue
    #         legals = self._KEY_INFOS_[key].choices
    #         if legals and value not in legals:
    #             sys.stderr.write("Arg checking failed: --{}={}\n".format(key, value))
    #             sys.stderr.write("Error: arg should be in {}\n".format("/".join(legals)))
    #             sys.stderr.flush()
    #             quit()


def get_comment(line):
    tokens = tokenize.tokenize(io.BytesIO(line.encode()).readline)
    tokstring = None
    for toktype, tokstring, _, _, _ in tokens:
        if toktype is tokenize.COMMENT:
            tokstring = tokstring[1:]
            break
    return tokstring


class ArgParser:
    def __init__(self):
        self.ap = argparse.ArgumentParser()

    def request(self, key, value):
        self.ap.add_argument('--{}'.format(key),
                             action='store',
                             default=value,
                             type=type(value),
                             dest=str(key))

    def parse(self):
        return self.ap.parse_args()
