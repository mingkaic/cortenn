
#ifndef DISABLE_OPMAP_TEST


#include "gtest/gtest.h"

#include "age/generated/opmap.hpp"


namespace age
{

#define _TYPE_EXEC_CALL(REAL_TYPE)typed_exec<REAL_TYPE>(out, opcode, shape, in);

void op_exec (_GENERATED_OPCODE opcode, _GENERATED_DTYPE dtype,
	SweetPotato& out, ade::Shape shape, Pomegranate& in)
{
	TYPE_LOOKUP(_TYPE_EXEC_CALL, dtype)
}

#undef _TYPE_EXEC_CALL

}


TEST(AGE, OpmapEminem)
{
	SweetPotato potato;
	Pomegranate pgram;
	Pomegranate pgram2;
	uint8_t dim = 32;
	ade::Shape shape({dim});

	age::op_exec(age::EMINEM, age::SAUSAGES, potato, shape, pgram);
	EXPECT_EQ(&pgram, potato.pptr_);
	EXPECT_EQ(dim, pgram.shape_.at(0));
	EXPECT_EQ(dim, pgram.shape_.n_elems());
	EXPECT_STREQ("meat", pgram.type_.c_str());
	EXPECT_EQ(Pomegranate::meat_hash(Meat(wraphash)), pgram.hash_);

	age::op_exec(age::EMINEM, age::HASHBROWNS, potato, shape, pgram2);
	EXPECT_EQ(&pgram2, potato.pptr_);
	EXPECT_EQ(dim, pgram2.shape_.at(0));
	EXPECT_EQ(dim, pgram2.shape_.n_elems());
	EXPECT_STREQ("fries", pgram2.type_.c_str());
	EXPECT_EQ(Pomegranate::fries_hash(Fries(wraphash)), pgram2.hash_);
}


TEST(AGE, OpmapKhaled)
{
	SweetPotato potato;
	Pomegranate pgram;
	Pomegranate pgram2;
	uint8_t dim = 76;
	ade::Shape shape({dim});

	age::op_exec(age::KHALED, age::SAUSAGES, potato, shape, pgram);
	EXPECT_EQ(&pgram, potato.pptr_);
	EXPECT_EQ(dim, pgram.shape_.at(0));
	EXPECT_EQ(dim, pgram.shape_.n_elems());
	EXPECT_STREQ("meat", pgram.type_.c_str());
	EXPECT_EQ(Pomegranate::meat_hash(Meat(djidx)), pgram.hash_);

	age::op_exec(age::KHALED, age::HASHBROWNS, potato, shape, pgram2);
	EXPECT_EQ(&pgram2, potato.pptr_);
	EXPECT_EQ(dim, pgram2.shape_.at(0));
	EXPECT_EQ(dim, pgram2.shape_.n_elems());
	EXPECT_STREQ("fries", pgram2.type_.c_str());
	EXPECT_EQ(Pomegranate::fries_hash(Fries(djidx)), pgram2.hash_);
}


#endif // DISABLE_OPMAP_TEST
