#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$script_dir"

mkdir -p repos
cd repos

clone_repo() {
	local url="$1"
	local target_name="${2:-}"

	if [[ -n "$target_name" ]]; then
		if [[ -d "$target_name/.git" || -d "$target_name" ]]; then
			echo "Skipping existing repo: $target_name"
			return 0
		fi
		git clone "$url" "$target_name"
		return 0
	fi

	local repo_name="${url##*/}"
	if [[ -d "$repo_name/.git" || -d "$repo_name" ]]; then
		echo "Skipping existing repo: $repo_name"
		return 0
	fi

	git clone "$url"
}

# TYPE A: Off-by-one
clone_repo https://github.com/PyCQA/isort
clone_repo https://github.com/encode/httpx
clone_repo https://github.com/arrow-py/arrow
clone_repo https://github.com/Delgan/loguru
clone_repo https://github.com/dry-python/returns

# TYPE B: Boolean flip
clone_repo https://github.com/Textualize/rich
clone_repo https://github.com/pyeve/cerberus
clone_repo https://github.com/python-attrs/cattrs
clone_repo https://github.com/python-attrs/attrs
clone_repo https://github.com/pallets/click

# TYPE C: Operator swap
clone_repo https://github.com/marshmallow-code/marshmallow
clone_repo https://github.com/seperman/deepdiff
clone_repo https://github.com/grantjenks/python-sortedcontainers sortedcontainers
clone_repo https://github.com/more-itertools/more-itertools
clone_repo https://github.com/mahmoud/glom

# TYPE D: Wrong variable
clone_repo https://github.com/fastapi/typer
clone_repo https://github.com/Suor/funcy
clone_repo https://github.com/dateutil/dateutil python-dateutil
clone_repo https://github.com/hynek/structlog
clone_repo https://github.com/agronholm/apscheduler

cd ..
