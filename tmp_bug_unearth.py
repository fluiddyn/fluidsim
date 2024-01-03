from datetime import datetime

from requests import Session

session = Session()

print("before get:", datetime.now())
resp = session.get(
    "https://pypi.org/simple/flit-core/",
    headers={
        "Accept": "application/vnd.pypi.simple.v1+json",
        "Cache-Control": "no-cache",
    },
    timeout=120,
)
print("after get:", datetime.now())

print(resp)
print(resp.content[-400:])
print(resp.json()["versions"])
