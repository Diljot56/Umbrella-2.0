import urllib.request
import json
import os

url = "https://gitlab.com/api/v4/projects/250833/issues?state=closed&per_page=5"
headers = {"User-Agent": "Mozilla/5.0"}
req = urllib.request.Request(url, headers=headers)

os.makedirs("data/raw/tickets", exist_ok=True)

with urllib.request.urlopen(req) as response:
    issues = json.loads(response.read().decode())

tickets = []
for item in issues:
    tickets.append({
        "ticket_id": f"GL-{item.get('iid')}",
        "title": item.get("title", ""),
        "created_at": item.get("created_at", ""),
        "url": item.get("web_url", ""),
        "thread": [
            {
                "author": (item.get("author") or {}).get("username", "user"),
                "body": item.get("description", "") or ""
            }
        ]
    })

output_path = "data/raw/tickets/runner_issues.json"
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(tickets, f, indent=2)

print(f"Successfully saved {len(tickets)} real GitLab tickets to {output_path}")
