COVERAGE_INFO_FILE := coverage.info

LLO_CTEST := //llo:ctest

LLO_PTEST := //llo:ptest

OPT_TEST := //opt:test

PBM_TEST := //pbm:test

TEST := bazel test

COVER := bazel coverage --config asan --config gtest

COVERAGE_IGNORE := 'external/*' '**/test/*' '**/genfiles/*'

COVERAGE_PIPE := ./scripts/merge_cov.sh $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/cortenn-test.log


benchmark:
	bazel run //llo:benchmark

coverage: cover_opt cover_llo cover_pbm

cover_llo:
	$(COVER) $(LLO_CTEST)

cover_opt:
	$(COVER) $(OPT_TEST)

cover_pbm:
	$(COVER) $(PBM_TEST)

# generated coverage files

lcov: coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/opt/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/llo/ctest/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/pbm/test/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_opt: cover_opt
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/opt/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_llo: cover_llo
	cat bazel-testlogs/llo/ctest/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'opt/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_pbm: cover_pbm
	cat bazel-testlogs/pbm/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)
