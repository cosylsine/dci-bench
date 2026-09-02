.PHONY: test prepare phase1-smoke phase2-toy phase3-one

TASK ?= LLMPublicHealthQA
QUERY_ID ?= Q25

test:
	python -m unittest discover -s tests -v

prepare:
	python -m dci_bench.data.workspace_builder --task $(TASK) --overwrite --sample-queries 5

phase1-smoke: prepare test

phase2-toy:
	python scripts/phase2_toy_smoke.py

phase3-one:
	python scripts/phase3_single_sample.py --task $(TASK) --query-id $(QUERY_ID) --prepare
