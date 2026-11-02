#!/usr/bin/env bash
# One-time setup of an Oracle Cloud Always Free VM (Ubuntu 22.04 or 24.04 on Ampere A1) for the ALFA app.
# Run it from this folder on the VM: bash setup.sh. It asks for your settings once and keeps them in .env and
# auth.env, then starts an empty copy of the app, so check_gate.sh can test sign-in before your data is served.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo usermod -aG docker "$USER"
fi

# Oracle's Ubuntu images reject inbound traffic in iptables even when the network's security list allows it
for port in 443; do
  sudo iptables -C INPUT -p tcp --dport "$port" -j ACCEPT 2>/dev/null \
    || sudo iptables -I INPUT 6 -m state --state NEW -p tcp --dport "$port" -j ACCEPT
done
sudo netfilter-persistent save

if [ ! -f .env ]; then
  read -rp "Your DuckDNS address (e.g. alfa-yourname.duckdns.org): " domain
  read -rp "Your Hugging Face username: " hf_user
  read -rp "A contact email for the news and price fetchers' user agent: " contact
  read -rsp "Your Hugging Face WRITE token (not shown): " hf_token; echo
  umask 077
  cat > .env <<SETTINGS
DOMAIN=$domain
HF_TOKEN=$hf_token
DATAFORGE_CONTACT=$contact
STOCKINTEL_MODEL_SPACE=$hf_user/stockintel-models
STOCKINTEL_SPACE_MODELS=writer
STOCKINTEL_CHRONOS_MODEL=$hf_user/stockintel-chronos2-nse
STOCKINTEL_KRONOS_MODEL=$hf_user/stockintel-kronos-nse
ALFA_MODELS_REPO=$hf_user/alfa-weights
ALFA_AGENT_REPO=$hf_user/alfa-agent
STOCKINTEL_STATE_REPO=$hf_user/stockintel-state
SETTINGS
  echo "Saved your settings to $(pwd)/.env (readable only by you)."
fi

# Its own file, so the other services never get the sign-in secrets; older installs get it too
if [ ! -f auth.env ]; then
  read -rp "Google OAuth client ID (Enter to leave Google sign-in off): " google_id
  google_secret=
  if [ -n "$google_id" ]; then read -rsp "Google OAuth client secret (not shown): " google_secret; echo; fi
  read -rp "SMTP host for sign-up emails (Enter to leave password sign-up off): " smtp_host
  smtp_port= smtp_user= smtp_password= smtp_from=
  if [ -n "$smtp_host" ]; then
    read -rp "SMTP SSL port (usually 465): " smtp_port
    read -rp "SMTP user name: " smtp_user
    read -rsp "SMTP password (not shown): " smtp_password; echo
    read -rp "Send emails from (e.g. StockIntel <you@yourdomain.com>): " smtp_from
  fi
  umask 077
  cat > auth.env <<SETTINGS
STOCKINTEL_AUTH_SECRET=$(openssl rand -hex 32)
GOOGLE_CLIENT_ID=$google_id
GOOGLE_CLIENT_SECRET=$google_secret
STOCKINTEL_SMTP_HOST=$smtp_host
STOCKINTEL_SMTP_PORT=$smtp_port
STOCKINTEL_SMTP_USER=$smtp_user
STOCKINTEL_SMTP_PASSWORD=$smtp_password
STOCKINTEL_SMTP_FROM=$smtp_from
SETTINGS
  echo "Saved your sign-in settings to $(pwd)/auth.env (readable only by you)."
fi

sudo docker compose -f compose.yaml -f gatecheck.yaml up -d --build
domain=$(grep '^DOMAIN=' .env | cut -d= -f2)
echo
echo "Started an empty copy of the app. Your data is not served yet. Next:"
echo "1. On your laptop, in this repository: bash deploy/oracle/check_gate.sh $domain"
echo "   It must pass. If any check fails, stop and fix it first."
echo "2. Here, serve your real data: sudo docker compose up -d"
echo "   Then run check_gate.sh again."
echo "3. Sign in on the site. If you used the app before sign-in, claim its portfolio:"
echo "   sudo docker compose exec app stockintel accounts claim --email <the email you signed in with>"
