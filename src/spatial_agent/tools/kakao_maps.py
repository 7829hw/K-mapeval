"""The slot `google_maps.py` occupied upstream.

`ecerybao/Spatial-Agent@6876bba` put its `GoogleMapsClient` here and imported it from
`agent/spatial_agent.py`, `agent/operators.py` and `agent/executors.py`. Those imports are
unchanged apart from the name; the client itself lives one level up, in `src/kakao_maps.py`,
because the vendored MapEval-API baseline reads the same one. Two architectures over one
evidence source is the experiment's whole premise, so there is exactly one implementation.
"""

from src.kakao_maps import KakaoMapsClient

__all__ = ["KakaoMapsClient"]
