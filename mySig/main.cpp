#pragma once
#include"siglink.h"
#include<functional>
#include<iostream>
struct A :mysig ::Object{
	void funcA(int a, int b) {
		std::cout <<"class in:" << a << " " << b << std::endl;
	}
};
void func_free(){}

int main() {
	using namespace mysig;
	A a;
	
	Base_Signal<int, int> base;
	connect(&base, &A::funcA, &a);
	connect(&base, [](int a,int b) {std::cout << "lambda in:" << a << " " << b << std::endl; });
	//base.emit(2, 3);
	//disconnect(&base, &A::funcA, &a);
	base.emit(2, 3);
	Base_Signal<int, int> base2;
	connect(&base2, &A::funcA, &a);
	
	//disconnect_all(&A::funcA, &a);
	printf("after disconnect_all\n");
	base.emit(2, 3);
}