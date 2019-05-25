
#ifndef DISABLE_CODES_TEST


#include "gtest/gtest.h"

#include "age/generated/codes.hpp"


TEST(AGE, Opcode)
{
	std::string djay = age::name_op(age::KHALED);
	std::string wrapper = age::name_op(age::EMINEM);

	EXPECT_STREQ("KHALED", djay.c_str());
	EXPECT_STREQ("EMINEM", wrapper.c_str());
	EXPECT_STREQ("BAD_OP", age::name_op(age::BAD_OP).c_str());

	EXPECT_EQ(age::KHALED, age::get_op(djay));
	EXPECT_EQ(age::EMINEM, age::get_op(wrapper));
	EXPECT_EQ(age::BAD_OP, age::get_op("KANYE"));
}


TEST(AGE, Typecode)
{
	std::string sausage = age::name_type(age::SAUSAGES);
	std::string hashs = age::name_type(age::HASHBROWNS);

	EXPECT_STREQ("SAUSAGES", sausage.c_str());
	EXPECT_STREQ("HASHBROWNS", hashs.c_str());
	EXPECT_STREQ("BAD_TYPE", age::name_type(age::BAD_TYPE).c_str());

	EXPECT_EQ(age::SAUSAGES, age::get_type(sausage));
	EXPECT_EQ(age::HASHBROWNS, age::get_type(hashs));
	EXPECT_EQ(age::BAD_TYPE, age::get_type("DRAKE"));

	EXPECT_EQ(age::SAUSAGES, age::get_type<Meat>());
	EXPECT_EQ(age::HASHBROWNS, age::get_type<Fries>());
	EXPECT_EQ(age::BAD_TYPE, age::get_type<std::string>());

	EXPECT_EQ(sizeof(Meat), age::type_size(age::SAUSAGES));
	EXPECT_EQ(sizeof(Fries), age::type_size(age::HASHBROWNS));
	EXPECT_THROW(age::type_size(age::BAD_TYPE), std::runtime_error);
}


#endif // DISABLE_CODES_TEST
