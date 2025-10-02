#!/usr/bin/bash

# Stop if any command fails.
set -e

# Change to the directory of this script so that relative paths resolve relative to this script.
cd "$(dirname "${BASH_SOURCE[0]}")"

# Agent environment.
docker push aisiuk/evals-cybench-agent-sandbox:1.0.0

# Challenge environments.
docker push aisiuk/evals-cybench-avatar-victim:1.0.0
docker push aisiuk/evals-cybench-back_to_the_past-victim:1.0.0
docker push aisiuk/evals-cybench-chunky-blog:1.0.0
docker push aisiuk/evals-cybench-chunky-nginx:1.0.0
docker push aisiuk/evals-cybench-chunky-rcache:1.0.0
docker push aisiuk/evals-cybench-delulu-victim:1.0.0
docker push aisiuk/evals-cybench-diffecient-victim:1.0.0
docker push aisiuk/evals-cybench-ezmaze-victim:1.0.0
docker push aisiuk/evals-cybench-failproof-victim:1.0.0
docker push aisiuk/evals-cybench-frog_waf-victim:1.0.0
docker push aisiuk/evals-cybench-flag_command-victim:1.0.0
docker push aisiuk/evals-cybench-glacier_exchange-victim:1.0.0
docker push aisiuk/evals-cybench-just_another_pickle_jail-victim:1.0.0
docker push aisiuk/evals-cybench-labyrinth_linguist-victim:1.0.0
docker push aisiuk/evals-cybench-locktalk-victim:1.0.0
docker push aisiuk/evals-cybench-motp-victim:1.0.0
docker push aisiuk/evals-cybench-network_tools-victim:1.0.0
docker push aisiuk/evals-cybench-noisier_crc-victim:1.0.0
docker push aisiuk/evals-cybench-noisy_crc-victim:1.0.0
docker push aisiuk/evals-cybench-randsubware-victim:1.0.0
docker push aisiuk/evals-cybench-robust_cbc-victim:1.0.0
docker push aisiuk/evals-cybench-shuffled_aes-victim:1.0.0
docker push aisiuk/evals-cybench-were_pickle_phreaks_revenge-victim:1.0.0
docker push aisiuk/evals-cybench-walking_to_the_sea_side-victim:1.0.0
docker push aisiuk/evals-cybench-unbreakable-victim:1.0.0
docker push aisiuk/evals-cybench-skilift-victim:1.0.0
