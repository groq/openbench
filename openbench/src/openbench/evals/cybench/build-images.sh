#!/usr/bin/env bash

# Stop if any command fails.
set -e

# Change to the directory of this script so that relative paths resolve relative to this script.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Agent environment.
docker build -t aisiuk/evals-cybench-agent-sandbox:1.0.0 ./agent_sandbox/

# Challenge environments.
docker build -t aisiuk/evals-cybench-avatar-victim:1.0.0 ./challenges/avatar/images/victim
docker build -t aisiuk/evals-cybench-back_to_the_past-victim:1.0.0 ./challenges/back_to_the_past/images/victim
docker build -t aisiuk/evals-cybench-chunky-blog:1.0.0 ./challenges/chunky/images/blog
docker build -t aisiuk/evals-cybench-chunky-nginx:1.0.0 ./challenges/chunky/images/nginx
docker build -t aisiuk/evals-cybench-chunky-rcache:1.0.0 ./challenges/chunky/images/rcache
docker build -t aisiuk/evals-cybench-delulu-victim:1.0.0 ./challenges/delulu/images/victim
docker build -t aisiuk/evals-cybench-diffecient-victim:1.0.0 ./challenges/diffecient/images/victim
docker build -t aisiuk/evals-cybench-ezmaze-victim:1.0.0 ./challenges/ezmaze/images/victim
docker build -t aisiuk/evals-cybench-failproof-victim:1.0.0 ./challenges/failproof/images/victim
docker build -t aisiuk/evals-cybench-frog_waf-victim:1.0.0 ./challenges/frog_waf/images/victim
docker build -t aisiuk/evals-cybench-flag_command-victim:1.0.0 ./challenges/flag_command/images/victim
docker build -t aisiuk/evals-cybench-glacier_exchange-victim:1.0.0 ./challenges/glacier_exchange/images/victim
docker build -t aisiuk/evals-cybench-just_another_pickle_jail-victim:1.0.0 ./challenges/just_another_pickle_jail/images/victim
docker build -t aisiuk/evals-cybench-labyrinth_linguist-victim:1.0.0 ./challenges/labyrinth_linguist/images/victim
docker build -t aisiuk/evals-cybench-locktalk-victim:1.0.0 ./challenges/locktalk/images/victim
docker build -t aisiuk/evals-cybench-motp-victim:1.0.0 ./challenges/motp/images/victim
docker build -t aisiuk/evals-cybench-network_tools-victim:1.0.0 ./challenges/network_tools/images/victim
docker build -t aisiuk/evals-cybench-noisier_crc-victim:1.0.0 ./challenges/noisier_crc/images/victim
docker build -t aisiuk/evals-cybench-noisy_crc-victim:1.0.0 ./challenges/noisy_crc/images/victim
docker build -t aisiuk/evals-cybench-randsubware-victim:1.0.0 ./challenges/randsubware/images/victim
docker build -t aisiuk/evals-cybench-robust_cbc-victim:1.0.0 ./challenges/robust_cbc/images/victim
docker build -t aisiuk/evals-cybench-shuffled_aes-victim:1.0.0 ./challenges/shuffled_aes/images/victim
docker build -t aisiuk/evals-cybench-were_pickle_phreaks_revenge-victim:1.0.0 ./challenges/were_pickle_phreaks_revenge/images/victim
docker build -t aisiuk/evals-cybench-walking_to_the_sea_side-victim:1.0.0 ./challenges/walking_to_the_sea_side/images/walking-to-the-sea-side
docker build -t aisiuk/evals-cybench-unbreakable-victim:1.0.0 ./challenges/unbreakable/images/victim
docker build -t aisiuk/evals-cybench-skilift-victim:1.0.0 ./challenges/skilift/images/victim
