# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

from .Strategy import Strategy

class BaselineStrategy(Strategy):
    def __init__(self, name="baseline"):
        super().__init__(name)
    

    