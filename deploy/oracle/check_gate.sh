#!/usr/bin/env bash
# Run from your laptop: bash deploy/oracle/check_gate.sh alfa-yourname.duckdns.org
# Passes only if the public host refuses anonymous callers with fastapi-users' own 401 and opens nothing but 443.
# Run it on setup.sh's empty copy before serving real data, and again after. It stops at the first failure.
set -euo pipefail
domain=${1:?give your DuckDNS address, e.g. bash check_gate.sh alfa-yourname.duckdns.org}
base="https://$domain"
UNAUTHORIZED='{"detail":"Unauthorized"}'

fail() { echo "FAIL: $*" >&2; exit 1; }

# expect <status> <body, or "" for any> <curl arguments...>
expect() {
  local status=$1 body=$2 answer
  shift 2
  answer=$(curl -sS --max-time 20 -w '\n%{http_code}' "$@") || fail "no answer to $*"
  [ "${answer##*$'\n'}" = "$status" ] || fail "$* answered ${answer##*$'\n'}, not $status: ${answer%$'\n'*}"
  [ -z "$body" ] || [ "${answer%$'\n'*}" = "$body" ] || fail "$* answered ${answer%$'\n'*}, not $body"
  echo "ok  $status  $*"
}

expect 200 "" "$base/health"
for path in /portfolio /portfolio/analysis /reports /users/me "/ui/search?q=a" /ui/stock/TCS/analyst; do
  expect 401 "$UNAUTHORIZED" "$base$path"
done
expect 401 "$UNAUTHORIZED" -X DELETE "$base/users/me"
expect 401 "$UNAUTHORIZED" -X POST -H "Content-Type: application/json" -d '{"message": "hi"}' "$base/chat"
expect 403 "" -H "Origin: https://evil.example" -d "username=gate-check@example.com&password=not-a-password" \
  "$base/auth/cookie/login"
headers=$(curl -sSI --max-time 20 "$base/health") || fail "no answer to $base/health"
for header in "content-security-policy: default-src" "strict-transport-security:"; do
  grep -qi "^$header" <<<"$headers" || fail "the proxy does not pass on the app's $header header"
  echo "ok  header  $header"
done
expect 404 "" "$base/openapi.json"
expect 404 "" "$base/docs"
for port in 80 8000 8010; do
  connected=$(curl -s -o /dev/null --connect-timeout 5 --max-time 10 -w '%{time_connect}' "http://$domain:$port/" || true)
  [ "$connected" = "0.000000" ] || fail "port $port accepts connections; the host must open only 443"
  echo "ok  closed  port $port"
done
echo "The sign-in gate holds."
