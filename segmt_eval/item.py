from dataclasses import dataclass
from typing import Union, List, Dict


@dataclass
class Item:
    item: str
    startOffSet: int
    endOffSet: int
    pos: str
    lemma: str
    isMinimumToken: bool
    isStopWord: bool
    ner: Union[str, List[Dict]]
