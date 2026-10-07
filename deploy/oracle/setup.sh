#!/usr/bin/env bash
# One-time setup of an Oracle Cloud Always Free VM (Ubuntu 22.04 or 24.04 on Ampere A1) for the ALFA app.
# Run it from this folder on the VM: bash setup.sh. It asks for your settings once and keeps them in .env.
set -euo pipefail
cd "$(dirname "$0")"

if ! command -v docker >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y docker.io docker-compose-v2
  sudo usermod -aG docker "$USER"
fi

# Oracle's Ubuntu images reject inbound traffic in iptables even when the network's security list allows it
for port in 80 443; do
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
STOCKINTEL_API_KEY=$(openssl rand -hex 24)
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

sudo docker compose up -d --build
domain=$(grep '^DOMAIN=' .env | cut -d= -f2)
key=$(grep '^STOCKINTEL_API_KEY=' .env | cut -d= -f2)
echo
echo "Started. The agent takes a few minutes to load. Your private login link:"
echo "  https://$domain/#key=$key"
echo "Keep it private: the key is the only login."
