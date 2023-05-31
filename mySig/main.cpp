#pragma once
#include"siglink.h"
#include<functional>
#include<iostream>
struct A {
	void funcA(int a, int b) {
		std::cout <<"class in:" << a << " " << b << std::endl;
	}
};
void func_free(){}

int main() {
	using namespace mysig;
	A a;
	Base_Signal<int, int> base;
	base.connect([](int a, int b) {std::cout << a << " " << b << std::endl; });
	base.connect([](int a, int b) {std::cout << a << " " << b << std::endl; });
	base.disconnect([](int a, int b) {std::cout << a << " " << b << std::endl; });
	base.connect(&A::funcA,&a);
	base.connect(&A::funcA, &a);
	base.disconnect(&A::funcA, &a);
	base.emit(3, 2);
}