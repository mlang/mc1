import json
import pytest
import subprocess

from mc1.dag import *

def roundtrip_json(graphFunc):
    return json.loads(
        subprocess.run([".build/default/dag2json"],
            capture_output=True, check=True, input=bytes(DAG(graphFunc))
        ).stdout
    )


@pytest.mark.parametrize("graphFunc, expected",
[( lambda freq=440: freq,
   {'name': '<lambda>',
    'constants': [],
    'controls': [440.0],
    'controlNames': [{'name': 'freq', 'index': 0}],
    'ops': [{'name': 'Control', 'rate': 98, 'num_out': 1, 'args': [0]}]
   }
 )
])
def test_roundtrip(graphFunc, expected):
    assert roundtrip_json(graphFunc) == pytest.approx(expected, rel=1e-6)
