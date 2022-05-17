

class Vocabulary(object):
    """ The vocabulary class maintains token to integer mapping in a bijective way. """

    def __init__(self, token_to_idx:dict = None, add_unk:bool = True, unk_token="<UNK>"):
        """

        :param token_to_idx (dict): a pre-existing map of tokens to indices
        :param add_unk (bool): a flag that indicates wheter to add the Unk Token
        :param unk_token (str): the Unk token to add into the vocabulary
        """

        if token_to_idx is None:
            token_to_idx = {}

        self._token_to_idx = token_to_idx
        self._idx_to_token = {idx: token
                              for token, idx in self._token_to_idx.items()}

        self._add_unk = add_unk
        self._unk_token = unk_token

        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token)


    def to_serializable(self):
        """ returns a dictionary that can be serialized """

        return {'token_to_idx': self._token_to_idx,
                'add_unk': self._add_unk,
                'un_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ instantiates the Vocabulary from a serialized dictionary """

        return cls(**contents)

    def add_token(self, token:str):
        """
        Update mapping dicts based on token

        :param token (str): The item to add into the Vocabulary
        :return (int): the id integer corresponding to the token
        """

        if token in self._token_to_idx:
            index = self._token_to_idx[token]
        else:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token

        return index

    def lookup_token(self, token):
        """
        retrieve the index associated with the token, or the Unk index if the token isn't present

        :param token (str): the token to lookup for
        :return (int): the index corresponding to the token
        """

        if self._add_unk:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]


    def lookup_index(self, index):
        """
        Return the token associated with the index

        :param index (int): The index to look up
        :return (str): the index corresponding to the index
        """

        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)

        return self._idx_to_token[index]

    def __str__(self):
        return "<voabulary(size==%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)