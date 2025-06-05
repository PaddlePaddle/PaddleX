from ...common.result import BaseResult


class Fastspeech2Result(BaseResult):

    def __init__(self, data: dict) -> None:
        super().__init__(data)