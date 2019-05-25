COVERAGE_INFO_FILE := coverage.info

BWD_TEST := //bwd:test

LLO_CTEST := //llo:ctest

TEST := bazel test

COVER := bazel coverage --config asan --config gtest

COVERAGE_IGNORE := 'external/*' '**/test/*' '**/genfiles/*'

COVERAGE_PIPE := ./bazel-bin/external/com_github_mingkaic_cppkg/merge_cov $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/cortenn-test.log


benchmark:
	bazel run //llo:benchmark

coverage: cover_bwd cover_llo

cover_bwd:
	$(COVER) $(BWD_TEST)

cover_llo:
	$(COVER) $(LLO_CTEST)

# generated coverage files

merge_cov:
	bazel build @com_github_mingkaic_cppkg//:merge_cov

lcov: merge_cov coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/llo/ctest/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_bwd: merge_cov cover_bwd
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/bwd/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'ade/*' -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_llo: merge_cov cover_llo
	cat bazel-testlogs/llo/ctest/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'opt/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)
