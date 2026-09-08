#!/usr/bin/env bash
set -euo pipefail
HOST=${HOST:-http://localhost:8000}
PROFILE=${1:-"I drive between sites all day, rarely near a charger, and I photograph equipment."}

jqp() { python3 -c 'import json,sys; d=json.load(sys.stdin); print(json.dumps(d, indent=2)[:1200])'; }

echo "== POST /threads"
RESP=$(curl -sS "$HOST/threads" -H 'content-type: application/json' -d "{\"profile\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$PROFILE")}")
echo "$RESP" | jqp
ID=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["thread_id"])')

while true; do
  DONE=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["done"])')
  [ "$DONE" = "True" ] && break
  ANSWER=$(echo "$RESP" | python3 -c 'import json,sys; print(json.load(sys.stdin)["ask_question"]["options"][0])')
  echo "== POST /threads/$ID/answer  <- $ANSWER"
  RESP=$(curl -sS "$HOST/threads/$ID/answer" -H 'content-type: application/json' \
    -d "{\"answer\": $(python3 -c 'import json,sys; print(json.dumps(sys.argv[1]))' "$ANSWER")}")
  echo "$RESP" | jqp
done

echo "== GET /threads/$ID"
curl -sS "$HOST/threads/$ID" | jqp
