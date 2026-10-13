import pytest

from rag_lab.business import Gateway
from rag_lab.fixtures import ALICE, AT, corpus
from rag_lab.service import Service


@pytest.fixture
def store():
    return corpus()


@pytest.fixture
def identity():
    return ALICE


@pytest.fixture
def at():
    return AT


@pytest.fixture
def gateway():
    return Gateway()


@pytest.fixture
def service(store, gateway):
    return Service(store, gateway)
