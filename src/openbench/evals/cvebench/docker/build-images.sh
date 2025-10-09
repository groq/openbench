#!/usr/bin/env bash

# Stop if any command fails.
set -e

commands=("pull" "build" "push")
command=$1

if [[ ! " ${commands[@]} " =~ " ${command} " ]]; then
  echo "Invalid command: $command"
  exit 1
fi

if [[ -n "$CVEBENCH_VERSION" && "$command" != "build" ]]; then
  echo "Not allowed to push or pull images when CVEBENCH_VERSION is set."
  exit 1
fi

# Change to the directory of this script so that relative paths resolve relative to this script.
cd "$(dirname "${BASH_SOURCE[0]}")"

function pull() {
  export PULL_POLICY=always
  docker compose pull
}

function build() {
  export PULL_POLICY=build
  docker compose build
}

function push() {
  before="$CVEBENCH_SANDBOX_COMPOSE_FILE"
  export CVEBENCH_SANDBOX_COMPOSE_FILE=../../docker/compose-empty.yml
  docker compose push
  export CVEBENCH_SANDBOX_COMPOSE_FILE="$before"
}

function run_cmd() {
  if [[ "$command" == "pull" ]]; then
    pull
  elif [[ "$command" == "build" ]]; then
    build
  elif [[ "$command" == "push" ]]; then
    push
  fi
}

run_cmd

for challenge in $(ls "${CVEBENCH_CHALLENGE_DIR:-../challenges}"); do
  pushd "${CVEBENCH_CHALLENGE_DIR:-../challenges}"/$challenge
  run_cmd
  popd
done
