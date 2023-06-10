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
	Base_Signal<int, int> base1;
	{
		A a;
		Base_Signal<int, int> base2;	
		connect(&base2, &A::funcA, &a);
		connect(&base1, &A::funcA, &a);
		base2.emit(2, 3);
		//disconnect_all(&A::funcA, &a);
	}
	base1.emit(1, 1);
	//connect(&base, [](int a,int b) {std::cout << "lambda in:" << a << " " << b << std::endl; });
	//base.emit(2, 3);
	//disconnect(&base, &A::funcA, &a);
	//base.emit(2, 3);
	//Base_Signal<int, int> base2;
	//connect(&base2, &A::funcA, &a);
	
	//printf("after disconnect_all\n");
	//base.emit(2, 3);
	//disconnect_all(&A::funcA, &a);
}