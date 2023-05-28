
#ifndef SIGLINK__
#define SIGLINK__
#include<vector>
#include<type_traits>
#include<utility>
#include<thread>
#include<mutex>
#include<functional>
#include<condition_variable>
#include<iostream>
#include<exception>
#include<unordered_map>
#include<unordered_set>
namespace mysig {
	/*Base class signal,*/

	/*
* Ҫ��1��ʹ��SFINAE���к���ģ���Ƶ�ʱ,���SFINAE��ν�ʣ��ɿ���ͨ������std::false_type������ʵ�֣�
* ͬʱ����β�ݹ�ķ�ʽ
* Ҫ��2��SFINAE����ģ���Ƶ�ʱ��ģ���������ģ���������ɣ���ò�ͬ����
* �ж�F(Args...)����ʱ��Ӧʹ��std::declval<Ty>()(std::declval<Args>()...��Ϊstd::declval<Ty>()
* ��ֱ�ӹ�����ֵ���ã������ô�������
* Ҫ��3������Ա��ģ����SFINAE����ʱ������class = std::enable_if_t<>ֻ�ܷ����������ģ���ͻ������
* �������Ϊ������ʾ������ģ�����
* �״�enable_if_t<>������յĵ�һ������Ϊbool���ͣ�����ֱ���������С����ʹ��ڡ��ж�
* ����������ж���Ҫ���� auto func () -> delctype(  ,  )
*/
/*
*  obj_function_traits����ƥ�����Ա����ָ�룬lambda���󣬵���ƥ�����ɺ���ָ��
* 
*
* �������ͣ�R--�������� C--�º�������  Args--�β��б�����
*/ 
	template<class F,class =void>
	struct obj_function_traits {
		using type = void;
		static constexpr bool value = false;
	};
	template <typename R, typename... Args>
	struct obj_function_traits<R(Args...)> {
		using type = void;
		static constexpr bool value = false;
	};
	
	template <typename R, typename C, typename... Args>
	struct obj_function_traits<R(C::*)(Args...) const>
	{
		using type = std::function<R(Args...)>;
		static constexpr bool value = true;
	};
	template <class R,class... Args>
	struct obj_function_traits<R(*)(Args...)> {
		using type = void;
		static constexpr bool value = false;
	};
	template <class F>
	struct obj_function_traits<F> : public obj_function_traits<decltype(&F::operator())> {};
	template<class F>
	using obj_function_type_t =typename obj_function_traits<F>::type;
	template<class F>
	constexpr bool is_obj_function_v = obj_function_traits<F>::value;
	template <typename F>
	obj_function_type_t<F> to_function(F& lambda)
	{
		return static_cast<obj_function_type_t<F>>(lambda);
	}
    template<class T>
	constexpr const char* class_tos(){
		return typeid(T).name();
	}

/*��Ҫ�õ�һ������������������ƥ��ʱ�����Բ���template<class Target, class First,class... Ty>
* �������������Ͳ�������ʽ
*/
	template<class ...>
	struct oneofAll{};
	template<class Target>
	struct oneofAll<Target> : std::false_type{};
	template<class Target, class First,class... Ty>
	struct oneofAll<Target,First,Ty...>:std::conditional_t < std::is_same_v< Target, First>, std::true_type, oneofAll<Target,Ty...>>{};
	template<class Target, class... Ty>
	constexpr bool is_oneofAll = oneofAll< Target, Ty...>::value;
	class MultiThreadPolicy {};
	template<class Ty, class... Args>
	class Callable {
	private:
		template<class Fun>
		static auto judge(int) -> decltype(std::declval<Fun>()(std::declval<Args>()...),std::true_type()){}
		template<class Fun>
		static auto  judge(...) -> decltype(std::false_type()) {}
	public:
		static constexpr  bool value = decltype(judge<Ty>(0))::value;
	};
	// ģ�����������constexpr ��������constexpr static
	template<class Ty, class... Args>
	constexpr bool is_Callable = Callable< Ty, Args...>::value;
/* TypeQueue���Ͷ��У����Զ��б������ͽ������²�����
* �������ף�������β��ѹ����ף�ѹ���β��Ѱ���±�Ϊk��Ԫ��
*/
	 template<class... Ty>
	 struct TypeQueue {};
	 template<class Queue>
	 struct EmptyType {};
	 //��ģ��Ĳ�����������ĩβ�����ػ��汾���ã�ע����ģ������Ͳ���ʵ�����ػ��汾��ʵ�ζ�Ӧ����
	 //�ػ��汾��ģ������������ֶ�Ӧ
	 template<class Queue,class Ele>
	 struct Pushback_TypeQueue{};
	 template<class... Ty,class Ele>
	 struct Pushback_TypeQueue<TypeQueue<Ty...>,Ele>{
		 using type = TypeQueue<Ty...,Ele>;
	 };
	 template<class Queue>
	 struct FrontType{};
	 template<class Ele, class... Ty>
	 struct FrontType<TypeQueue<Ele, Ty...>> {
		 using type = Ele;
	 };
	 // ��������Ͷ����ж���Ԫ��
	 template<>
	 struct FrontType<TypeQueue<>> {
		 using type = EmptyType<TypeQueue<>>;
	 };
	 template<class Queue>
	 struct PopFrontType{};
	 template<class Ele, class... Ty>
	 struct PopFrontType<TypeQueue<Ele, Ty...>> {
		 using type = TypeQueue<Ty...>;
	 };
	 
	template<class First,class... Left>
	struct EmptyType<TypeQueue<First, Left...>>{
		using type = First;
		static constexpr bool value = false;
	};
	// ������û��type��Ա
	template<>
	struct EmptyType<TypeQueue<>> {
		static constexpr bool value = true;
	};
	
	template<class Queue>
	constexpr bool is_empty_v = EmptyType<Queue>::value;
	/*QueueCompertor< QueueLeft, QueueRight,Operation>��QueuLeft��QueuRight��ÿ��Ԫ��
	* ��������ͣ���QueuLeft��ÿ��Ԫ����QueuRight��ͬ����valueΪ1��ͬʱ�ñȽ�������ȡ
	* ���ͣ���QueuLeft��ÿ��Ԫ����QueuRight��ͬ��������Ϊis_empty<>,����Ϊ��һ������ͬ
	* ��Ԫ�ض� QueueType<L,R>
	*/
	 template<class QueueLeft,class QueueRight,template<class T,class U>class Operation,
	 bool empty = is_empty_v<QueueLeft>|| is_empty_v<QueueRight>>
	 struct QueueComperator;
	 template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	 struct QueueComperator<QueueLeft, QueueRight, Operation,false>{
		 static constexpr bool value = Operation<typename FrontType< QueueLeft>::type,
			 typename FrontType< QueueRight>::type>::value && QueueComperator<typename PopFrontType<QueueLeft>::type,
			 typename PopFrontType<QueueRight>::type, Operation>::value;
		 using type = std::conditional_t< !Operation<typename FrontType< QueueLeft>::type,
			 typename FrontType< QueueRight>::type>::value, TypeQueue<typename FrontType< QueueLeft>::type,
			 typename FrontType< QueueRight>::type>, typename  QueueComperator<typename PopFrontType<QueueLeft>::type,
			 typename PopFrontType<QueueRight>::type, Operation>::type>;
	 };
	 template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	 struct QueueComperator<QueueLeft, QueueRight, Operation, true> {
		 static constexpr bool value = is_empty_v< QueueLeft> ;
		 using type = std::conditional_t<is_empty_v< QueueLeft> && is_empty_v< QueueRight>, EmptyType<QueueLeft>,
		     std::conditional_t<is_empty_v< QueueLeft>,TypeQueue<typename FrontType<QueueRight>::type>, 
		     TypeQueue<typename FrontType<QueueLeft>::type>>>;
	 };
	 template<class T,class U>
	 struct equal_operation {
		 static constexpr bool value = std::is_same_v<T, U>;
	 };
	 template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	 constexpr bool QueueCompertor_v = QueueComperator<QueueLeft, QueueRight, Operation>::value;
	 template<class QueueLeft, class QueueRight>
	 constexpr bool is_QueueType_include = QueueCompertor_v< QueueLeft, QueueRight, equal_operation>;
	 template<class QueueLeft, class QueueRight>
	 constexpr bool is_QueueType_same = is_QueueType_include<QueueLeft, QueueRight> && is_QueueType_include<QueueRight, QueueLeft>;
/*��ȡ�ɵ��ö�����β��б�����,�ɵ��ö���F��operator()���� R(C::*)(Args...)��ʽ
* ��������ΪTypeQueue<Args...>
 */
	 template<class T, class = void>
	 struct function_Args {};
	 template<class T>
	 struct function_Args<T> {
		 using type = typename function_Args<decltype(&T::operator())>::type;
	 };
	 template<class R, class T, class... Args>
	 struct function_Args<R(T::*)(Args...) const> {//lambda�����operator()����Ϊconst����
		 using type = TypeQueue<Args...>;
	 };
	 template<class R,class... Args>
	 struct function_Args<R(Args...)> {
		 using type = TypeQueue<Args...>;
	 };
	 template<class T>
	 using function_Args_t = typename function_Args<T>::type;

	class Base_Function {
	public:
		Base_Function() = default;
		template<class F>
		Base_Function(F&& f){}
		Base_Function(const Base_Function&) = delete;
		Base_Function& operator=(const Base_Function&) = delete;
		Base_Function(Base_Function&&) noexcept;
		Base_Function& operator=(Base_Function&&) noexcept{}
		virtual ~Base_Function() {}
	};
	template<class F, class Queue>
	class Concert_Function;
	template<class F,class... Args>
	class Concert_Function<F, TypeQueue<Args...>> :public Base_Function {
	public:
		using Args_type = TypeQueue< Args...>;
		//F should be std::function<void(CallArgs...)>
		Concert_Function(F f) : func(f) {}
		template<class... CallArgs>
		void invoke(CallArgs&& ... args) {
			if (!this) {
				throw std::invalid_argument("invoke agrs must match with agrs of callalbe object");
			}
			 func(std::forward<CallArgs>(args)...);
		}
		~Concert_Function() override{}
	private:
		F func;
	};
	/* ���Ͳ������
	** ������FunctionWrapper()
	*/
	class FunctionWrapper {
	public:
		FunctionWrapper():func_ptr(nullptr){}
		template<class F>
		FunctionWrapper(F&& func):func_ptr(new Concert_Function < obj_function_type_t
			<std::decay_t<F>>,function_Args_t<std::decay_t<F>>>(std::forward<F>(func))) {
			std::cout << "F������" << typeid(decltype(*func_ptr)).name() << std::endl;
		}
		FunctionWrapper(const FunctionWrapper&) = delete;
		FunctionWrapper& operator=(const FunctionWrapper&) = delete;
		FunctionWrapper( FunctionWrapper && wp) noexcept {
			func_ptr.reset(wp.func_ptr.get());
			wp.func_ptr.release();
		}
		FunctionWrapper& operator=(FunctionWrapper&& wp) noexcept {
			func_ptr.reset(wp.func_ptr.get());
			wp.func_ptr.release();
			return *this;
		}
		std::unique_ptr< Base_Function> func_ptr;
	};
	/* ���ܺ���������,���������͵Ŀɵ��ö���������Ͳ���
	* ���÷�Χ��std::funcion<T> ������ֵlambda�������Ա������ ��ֹƥ�䷶Χ��ȫ�ֺ���ָ��
	* �ӿ� @UniversalFunction(std::funcion<T>(...));
	*    UniversalFunction f1([](int,int){});
	*    UniversalFunction f2(std::function<void(long)>);
	*    UniversalFunction f3(&A::func);
	*    f1(1,1); f2(2); f3();
	*
	*/
	class UniversalFunction {		
	public:
		UniversalFunction() = default;
		//ensure copy and move construct function have higher priority than it
		template <class Func, class Check = std::enable_if_t<is_obj_function_v<std::decay_t<Func>>> >
		UniversalFunction(Func&& func):wp(std::forward<Func>(func)){}
		template <class R,class F,class... Args>
		UniversalFunction(R(F::* func)(Args...), F* t):
		wp([this,func,t](Args... args){ (t->*func)(std::forward< Args>(args)...);  }) {}
		UniversalFunction(const UniversalFunction&) = delete;
		UniversalFunction& operator=(const UniversalFunction&) = delete;
		UniversalFunction(UniversalFunction&& f):wp(std::move(f.wp)) {}
		UniversalFunction& operator=(UniversalFunction&& f) {
			wp = std::move(f.wp);
			return *this;
		}
		template<class... CallArgs>
		void operator()(CallArgs&& ...args) {
			try {
			    (this->function_cast<std::function<void(CallArgs...)>>())->invoke(std::forward<CallArgs>(args)...);
			}
			catch (std::exception& e) {
				std::cerr << e.what() << std::endl;
			}
		}
		template<class T>
		decltype(auto) function_cast() {
			return (dynamic_cast<Concert_Function<T, function_Args_t<std::decay_t<T>>>*>(wp.func_ptr.get()));
		}
	private:
		FunctionWrapper wp;
	};
	template< class... Args> class Signal {};
	template<class Sender,class... SigArgs,class Receiver,class F>
	void connect(Sender* sender, Signal<SigArgs...> Sender::* sigfunc, Receiver* receiver, F&& f) {
		static_assert(is_obj_function_v<std::decay_t<F>>, " F must be a member function pointer or lambda expression");
		static_assert(is_QueueType_same<TypeQueue<SigArgs...>,function_Args_t<std::remove_reference_t<F>>>, "SigArgs must have same Type with ReArgs");
		auto rfunc = to_function((f));
	
		//����ۺ���Ϊ��ֵ��ʱ��������ʱ
		if constexpr (std::is_reference_v<F>) {
			std::cout << "��ֵ����" << std::endl;
		}
		else {
			auto rfunc = to_function(f);
			std::cout << "��ʱ����" << std::endl;
		}
			
	}
	//using Slot = UniversalFunction;
	template<class Sender, class... SigArgs, class Receiver, class R, class... ReArgs>
	void connect(Sender* sender, Signal<SigArgs...> Sender::* sigfunc, Receiver* receiver, R(Receiver::* refunc)(ReArgs...) ) {
			std::function<R(ReArgs...)> rfunc;
	}
namespace detail {
	enum SIGPolicy {
		LINEAR,
		THREADPOOL,

	};
	template<class... Args>
	class Base_Slot {
	public:
		template<class... F>
		Base_Slot(F&&... func):uf(std::forward<F>()){}
		Base_Slot(const Slot&s):uf(s.uf){}
		Base_Slot( Slot&& s):uf(std::move(s.uf)){}
		Base_Slot& operator=(const Slot& s) {
			uf = s.uf;
			return *this;
		}
		virtual void seek();
	protected:
		UniversalFunction  uf;
		std::unordered_set<Base_Signal<Args...>*> sig_set;
	};
	template<class R,class C,class... Args>
	class Slot : public Base_Slot<Args...> {
		using func_type = R(C::*)(Args...);
		using return_type = R;
	public:
		void seek() override{

		}
	private:
		C* obj_ptr;
		func_type func;
	};

	template<class R,class C,class... Args>
	class SlotManeger {
	public:
	private:
		std::unordered_set<C*,R()
	};
	class ThreadPolicy;
	template<class... Args>
	class Base_Signal {
	public:
		using Signal_Type = TypeQueue<Args...>;
		Base_Signal(ThreadPolicy p) = default;
		template<class... F>
		void store(F&&... func) {
			std::unique_lock<std::mutex> lock(mut);
			sig_con.emplace_back(std::make_shared<Slot>( std::forward<F>(func)...));
			lock.unlock();
		}
		template<class... CallArgs>
		void signal(CallArgs&&... args) {
			std::unique_lock<std::mutex> lock(mut);
			for (auto&& sig : sig_con) {
				lock.lock();
				sig(std::forward<CallArgs>(args)...);
				lock.unlock();
			}
		}
	private:	
		std::unordered_set<std::unique_ptr<Base_Slot>> sig_con;
		std::mutex mut;
		ThreadPolicy tp;
	};
};
	//template<class ThreadPolicy,class R, class... Args>
	//class Signal : public Base_Signal<R,Args...> {
	//	
	//};
	/*
	template<class Ty,class... Args>
	class Base_signal {
		using ThreadPolicy = Ty;
	public:
		Base_signal() = default;
		template< class U, class... Args,class = std::enable_if_t< Callable<Func, Args...>::is_callable>>
		Base_signal(U &&s){}

		template<class U, class = std::enable_if_t<is_obj_function_v<std::remove_reference_t<U>>>>
		void connect(U&& func) {
			static_assert(std::is_same_v<function_type_t<std::remove_reference_t<U>>, Func >, "para must has same ");
			static_assert(is_oneofAll<Ty, int>,"wrong type");
			_connect(func);
		}
		template<class... Args,class = std::enable_if_t<is_Callable<Func,Args...>>>
		void signal(Args&&... args) {
			ThreadPolicy policy;
			std::unique_lock<std::mutex>  lock(mut);
			auto calls = callbacks.back();
			callbacks.pop_back();
			lock.unlock();
			calls(std::forward<Args>(args)...);
		}
	private:
		template<class U>
		void _connect(U&& func){
			std::lock_guard<std::mutex>  lock(mut);
			callbacks.push_back(std::forward<U>(func));
		}
		std::mutex mut;
		std::vector< Func> callbacks;
	};

	template< class Func>
	using signal = Base_signal < MultiThreadPolicy, Func>;
	*/
}

#endif
