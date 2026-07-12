from __future__ import annotations

import csv
import json
from pathlib import Path

import cdsapi


REQUEST_LOG = Path(
    r"E:\research\垂直交换\上海_ERA5_2025_pressure_levels"
    r"\era5_2025_shanghai_pressure_levels_requests.csv"
)


def main() -> None:
    rows = list(csv.DictReader(REQUEST_LOG.open("r", encoding="utf-8-sig", newline="")))
    client = cdsapi.Client(wait_until_complete=False, quiet=True)

    print("period,status_in_log,remote_state,request_id")
    for row in rows:
        request_id = row.get("request_id", "").strip()
        if not request_id:
            print(f"{row['period']},{row['status']},no_request_id,")
            continue

        reply_json = row.get("reply_json", "").strip()
        if reply_json:
            reply = json.loads(reply_json)
            links = reply.get("links", [])
            self_links = [link["href"] for link in links if link.get("rel") == "self"]
            url = self_links[0] if self_links else f"{client.url}/retrieve/v1/jobs/{request_id}"
        else:
            url = f"{client.url}/retrieve/v1/jobs/{request_id}"
        response = client.session.get(url, verify=client.verify, timeout=60)
        if response.status_code != 200:
            print(f"{row['period']},{row['status']},HTTP_{response.status_code},{request_id}")
            continue

        reply = response.json()
        state = reply.get("state") or reply.get("status") or ""
        print(f"{row['period']},{row['status']},{state},{request_id}")


if __name__ == "__main__":
    main()
