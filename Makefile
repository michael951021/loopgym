SHELL := /bin/bash
PY := $(CURDIR)/.venv/bin/python
CARGO_ENV := CARGO_HOME=$(CURDIR)/.tools/cargo RUSTUP_HOME=$(CURDIR)/.tools/rust PATH=$(CURDIR)/.tools/cargo/bin:$(CURDIR)/.tools/bin:$(PATH)

.PHONY: bootstrap check known

bootstrap:
	bash tools/bootstrap.sh

check:
	$(PY) -m ruff check .
	$(PY) -m pytest -q tests --ignore=tests/known
	@if [ -d lgbox ]; then \
	  echo "cargo test (lgbox)"; \
	  $(CARGO_ENV) cargo test --manifest-path lgbox/Cargo.toml; \
	else echo "skip cargo test: lgbox/ not present yet"; fi
	@sh_files=$$(find tools scripts deploy -type f -name '*.sh' 2>/dev/null); \
	if [ -n "$$sh_files" ]; then \
	  echo "shellcheck"; $(CURDIR)/.tools/bin/shellcheck $$sh_files; \
	else echo "skip shellcheck: no .sh files under tools/ scripts/ deploy/ yet"; fi
	@docks=$$(find deploy -type f -name 'Dockerfile*' 2>/dev/null); \
	if [ -n "$$docks" ]; then \
	  echo "hadolint"; $(CURDIR)/.tools/bin/hadolint $$docks; \
	else echo "skip hadolint: no Dockerfiles under deploy/ yet"; fi
	@if [ -d deploy/aws ] && ls deploy/aws/*.tf >/dev/null 2>&1; then \
	  echo "terraform validate (deploy/aws, -backend=false only)"; \
	  $(CURDIR)/.tools/bin/terraform -chdir=deploy/aws init -backend=false -input=false; \
	  $(CURDIR)/.tools/bin/terraform -chdir=deploy/aws validate; \
	else echo "skip terraform: deploy/aws not present yet"; fi

known:
	bash tools/check_known.sh
	LOOPGYM_KNOWN_STRICT=1 $(PY) -m pytest -q tests/known
