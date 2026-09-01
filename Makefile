.PHONY: test prepare phase1-smoke

TASK ?= LLMPublicHealthQA

test:
	python -m unittest discover -s tests -v

prepare:
	python -m dci_bench.data.workspace_builder --task $(TASK) --overwrite --sample-queries 5

phase1-smoke: prepare test
