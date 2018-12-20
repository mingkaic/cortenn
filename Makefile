COVERAGE_INFO_FILE := coverage.info

LLO_CTEST := //llo:ctest

LLO_PTEST := //llo:ptest

REGRESS_TEST := //llo_regress:test_regress

PBM_TEST := //pbm:test

TEST := bazel test

COVER := bazel cover

C_FLAGS := --config asan --config gtest

COVERAGE_IGNORE := 'external/*' '**/test/*' 'testutil/*' '**/genfiles/*' 'dbg/*'

COVERAGE_PIPE := ./scripts/merge_cov.sh $(COVERAGE_INFO_FILE)

TMP_LOGFILE := /tmp/cortenn-test.log


test: test_cllo test_pyllo test_pbm

test_cllo:
	$(TEST) $(C_FLAGS) --config grepeat $(LLO_CTEST)

test_pyllo:
	$(TEST) $(LLO_PTEST)

test_pbm:
	$(TEST) $(C_FLAGS) $(PBM_TEST)


coverage: cover_llo cover_pbm

cover_llo:
	$(COVER) $(C_FLAGS) --config grepeat $(LLO_CTEST)

cover_pbm:
	$(COVER) $(C_FLAGS) $(PBM_TEST)

# generated coverage files

lcov_all: coverage
	rm -f $(TMP_LOGFILE)
	cat bazel-testlogs/llo/test/test.log >> $(TMP_LOGFILE)
	cat bazel-testlogs/pbm/test/test.log >> $(TMP_LOGFILE)
	cat $(TMP_LOGFILE) | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) -o $(COVERAGE_INFO_FILE)
	rm -f $(TMP_LOGFILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_llo: cover_llo
	cat bazel-testlogs/llo/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' 'age/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

lcov_pbm: cover_pbm
	cat bazel-testlogs/pbm/test/test.log | $(COVERAGE_PIPE)
	lcov --remove $(COVERAGE_INFO_FILE) $(COVERAGE_IGNORE) 'log/*' 'ade/*' 'age/*' 'llo/*' -o $(COVERAGE_INFO_FILE)
	lcov --list $(COVERAGE_INFO_FILE)

# test management

dora_run:
	./scripts/start_dora.sh ./certs

gen_test: dora_run
	bazel run //test_gen:tfgen

test_regress: gen_test
	$(TEST) $(C_FLAGS) $(REGRESS_TEST)
