#pragma once
#include"siglink.h"
#include<functional>
#include<iostream>
struct A {
	mysig::Signal<> s;
};
void funcB(){}
struct B {
	void funcB(int layer) { 
		auto aa = [this](int l) {
			std::cout << "funcB" << l << std::endl;
			if (l < 5)
				funcB(l + 1);
		};
		mysig::UniversalFunction f(aa);
		f((int)layer);
	}
};
void test_Lists() {
	std::cout <<"int float char:"<< mysig::QueueCompertor_v< mysig::TypeQueue<int, float, char>, mysig::TypeQueue<int, float, char>, mysig::equal_operation> << std::endl;//1
	std::cout << "int float char:" << mysig::QueueCompertor_v< mysig::TypeQueue<int, float, char>, mysig::TypeQueue<int, float, int>, mysig::equal_operation> << std::endl;//0
	std::cout << "int float char:" << mysig::QueueCompertor_v< mysig::TypeQueue<int, float, char>, mysig::TypeQueue<int, float, int,void>, mysig::equal_operation> << std::endl;//0
	std::cout << "int float char:" << mysig::QueueCompertor_v< mysig::TypeQueue<int, float, decltype(&B::funcB)>, mysig::TypeQueue<int, float, decltype(&B::funcB)>, mysig::equal_operation> << std::endl;//1
}
void UniversalFunctiontest() {
	using namespace mysig;
}
int main() {
	using namespace mysig;
	test_Lists();
	A a_obj; B b_obj;
	auto aa = [&]() {std::cout << "22" << std::endl; };
	connect(&a_obj, &A::s, &b_obj, &B::funcB);
	connect(&a_obj, &A::s, &b_obj, [&]() {});
	connect(&a_obj, &A::s, &b_obj, aa);
	std::cout << "begin test UniversalFunction" << std::endl;
	UniversalFunction f1([](int a, int  b) {std::cout <<"f1" << "a=" << a << "b=" << b << std::endl; });
	UniversalFunction f2(&B::funcB,&b_obj);
	UniversalFunction f3([](int a, int&  b) {std::cout <<"f3" << "a=" << a << "b=" << b << std::endl; });
	int kk = 3;
	f1(2, 3);
	UniversalFunction f4(std::move(f1));
	f1(2, 3);
	f4(2, 3);
	f2(0);
}