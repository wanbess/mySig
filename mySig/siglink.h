
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
#include<list>
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
	template<class F, class = void>
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
	template <class R, class... Args>
	struct obj_function_traits<R(*)(Args...)> {
		using type = void;
		static constexpr bool value = false;
	};
	template <class F>
	struct obj_function_traits<F> : public obj_function_traits<decltype(&F::operator())> {};
	template<class F>
	using obj_function_type_t = typename obj_function_traits<F>::type;
	template<class F>
	constexpr bool is_obj_function_v = obj_function_traits<F>::value;
	template <typename F>
	obj_function_type_t<F> to_function(F& lambda)
	{
		return static_cast<obj_function_type_t<F>>(lambda);
	}
	template<class T>
	constexpr const char* class_tos() {
		return typeid(T).name();
	}

	/*��Ҫ�õ�һ������������������ƥ��ʱ�����Բ���template<class Target, class First,class... Ty>
	* �������������Ͳ�������ʽ
	*/
	template<class ...>
	struct oneofAll {};
	template<class Target>
	struct oneofAll<Target> : std::false_type {};
	template<class Target, class First, class... Ty>
	struct oneofAll<Target, First, Ty...> :std::conditional_t < std::is_same_v< Target, First>, std::true_type, oneofAll<Target, Ty...>> {};
	template<class Target, class... Ty>
	constexpr bool is_oneofAll = oneofAll< Target, Ty...>::value;
	class MultiThreadPolicy {};
	template<class Ty, class... Args>
	class Callable {
	private:
		template<class Fun>
		static auto judge(int) -> decltype(std::declval<Fun>()(std::declval<Args>()...), std::true_type()) {}
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
	template<class Queue, class Ele>
	struct Pushback_TypeQueue {};
	template<class... Ty, class Ele>
	struct Pushback_TypeQueue<TypeQueue<Ty...>, Ele> {
		using type = TypeQueue<Ty..., Ele>;
	};
	template<class Queue>
	struct FrontType {};
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
	struct PopFrontType {};
	template<class Ele, class... Ty>
	struct PopFrontType<TypeQueue<Ele, Ty...>> {
		using type = TypeQueue<Ty...>;
	};

	template<class First, class... Left>
	struct EmptyType<TypeQueue<First, Left...>> {
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
	template<class QueueLeft, class QueueRight, template<class T, class U>class Operation,
		bool empty = is_empty_v<QueueLeft> || is_empty_v<QueueRight>>
		struct QueueComperator;
	template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	struct QueueComperator<QueueLeft, QueueRight, Operation, false> {
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
		static constexpr bool value = is_empty_v< QueueLeft>;
		using type = std::conditional_t<is_empty_v< QueueLeft>&& is_empty_v< QueueRight>, EmptyType<QueueLeft>,
			std::conditional_t<is_empty_v< QueueLeft>, TypeQueue<typename FrontType<QueueRight>::type>,
			TypeQueue<typename FrontType<QueueLeft>::type>>>;
	};
	template<class T, class U>
	struct equal_operation {
		static constexpr bool value = std::is_same_v<T, U>;
	};
	template<class T, class U>
	struct convert_operation {//�ж�����U�Ƿ���ת����T��ע�����ﲻ�ܽ���˳��
		static constexpr bool value = std::is_convertible_v<U, T>;
	};
	template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	constexpr bool QueueCompertor_v = QueueComperator<QueueLeft, QueueRight, Operation>::value;
	template<class QueueLeft, class QueueRight>
	constexpr bool is_QueueType_include = QueueCompertor_v< QueueLeft, QueueRight, convert_operation>;
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
	template<class R, class... Args>
	struct function_Args<R(Args...)> {
		using type = TypeQueue<Args...>;
	};
	template<class T>
	using function_Args_t = typename function_Args<T>::type;
	/*�����ü����
	* ʹ�� trim_call()���вü�����
	*/
	template<class Fun,class... CallArgs>
	struct ParamTrim{
		ParamTrim (Fun&& f, CallArgs&&... args):
			fun(std::forward<Fun>(f)),t(std::forward<CallArgs>(args)...){}
		template<std::size_t... Index>
		void operator ()(std::index_sequence<Index...>) {
			fun(std::get<Index>(std::move(t))...);
		}
		template<class... Args>
		static constexpr bool checkType() {
			//�ж�CallArgs...�еĲ����ܷ���ʽת��ΪArgs...
			return is_QueueType_include<TypeQueue<Args...>, TypeQueue<CallArgs...>>;
		}
		std::tuple<CallArgs...> t;
		Fun fun;
		template<class... Args>
		static constexpr bool value = ParamTrim::checkType<Args...>();
	};
	template<class Fun,class... Args,class... CallArgs>
	void trim_call(Fun&& f, const TypeQueue<Args...>&, CallArgs&&... callargs) {
		using ParamTrim_t = ParamTrim< Fun, CallArgs...>;
		ParamTrim_t pt(std::forward<Fun>(f),std::forward<CallArgs>(callargs)...);
		static_assert(ParamTrim_t::template value<Args...>, "CallArgs can't be conver to Args or has different length");
		pt(std::index_sequence_for<Args...>{});
	}
	template<class... Args>
	class EventWrapper;
	enum class Func_Type {
		Lambda=0,
		MenberFunction=1,
		GlobalFunction=2,
		None=3
	};
	template<class... Args>
	class Base_Function {
	public:
		using addr_type = typename std::uintptr_t;//���ͱ���Ҫд��public�У�����Ĭ��private
		Base_Function():ft(Func_Type::None){}
		explicit Base_Function(const Func_Type& f):ft(f){}
		Base_Function(const Base_Function&) = delete;
		Base_Function& operator=(const Base_Function&) = delete;
		virtual void operator()(Args... args){}
		virtual std::vector<addr_type> get_address() {
			return  {  reinterpret_cast<addr_type>(this) };
		};
		Func_Type ft;
	protected:
		using Arg_types = TypeQueue<Args...>;
		
	};
	template<class Func,class... Args>
	class lambda_Function : public Base_Function<Args...> {
	public:
		using Base_type = typename Base_Function<Args...>;
		using Base_type::addr_type;
		lambda_Function(Func&& f):Base_Function<Args...>(Func_Type::Lambda),func(std::forward<Func>(f)){}
		void operator()(Args... args) override {
			func(std::forward<Args>(args)...);
		}
		virtual std::vector<addr_type> get_address() override{
			return  { Base_type::get_address()[0],reinterpret_cast<addr_type>(std::addressof<Func>(func)) };
		};
	private:		
		Func func;
	};
	template<class R,class C,class... Args>
	class Member_Funcion :public  Base_Function<Args...> {
	public:
		using Base_type = Base_Function<Args...>;
		using Base_type::addr_type;
		Member_Funcion(R(C::*f)(Args...) , C* obj):Base_Function<Args...>(Func_Type::MenberFunction), obj_ptr(obj),func_ptr(f){}
		void operator()(Args... args) override {
			(obj_ptr->*func_ptr)(std::forward<Args>(args)...);
		}
		virtual std::vector<addr_type> get_address() override {
			addr_type func_ptr_r;
			std::memcpy(&func_ptr_r, &func_ptr, sizeof(func_ptr));
			return  { Base_type::get_address()[0], 
				reinterpret_cast<addr_type>(obj_ptr),
			func_ptr_r };
		};
		
	private:
		C* obj_ptr;
		R(C::* func_ptr)(Args...) ;
	};
	template<class R, class... Args>
	class Global_Function :public  Base_Function<Args...> {
	public:
		using Base_type = Base_Function<Args...>;
		using Base_type::addr_type;
		Global_Function(R(*f)(Args...) ):Base_Function<Args...>(Func_Type::GlobalFunction), func_ptr(f){}
		void operator()(Args... args) override {
			(*func_ptr)(args...);
		}
		virtual std::vector<addr_type> get_address() override {
			return  { Base_type::get_address()[0],reinterpret_cast<addr_type>(func_ptr) };
		};
	private:	
		R(*func_ptr)(Args...) ;
	};
	template<class... Args>
	class EventWrapper {
	public:
		friend class EventWrapper<Args...>;
		EventWrapper(): func(nullptr){}
		template<class F, class Func_Check = std::enable_if_t<is_obj_function_v<std::decay_t<F>>>,
			class Args_Check = std::enable_if_t<std::is_same_v<TypeQueue<Args...>, function_Args_t<std::decay_t<F>>>>>
				EventWrapper(F&& f) {
				using Func_t = std::decay_t<F>;
				using lambda_t = lambda_Function<Func_t, Args...>;
				func.reset(new lambda_t(std::forward<F>(f)));
			}
		template<class R,class C>
		EventWrapper(R(C::* func_ptr)(Args...),C* obj):func(new Member_Funcion<R,C,Args...>(func_ptr,obj)){}
		template<class R>
		EventWrapper(R(* func_ptr)(Args...)) : func(new Global_Function<R,Args...>(func_ptr)) {}
		EventWrapper(const EventWrapper&) = delete;
		EventWrapper& operator=(const EventWrapper&) = delete;
		EventWrapper(EventWrapper&& w):func(std::move(w.func)){}
		EventWrapper& operator=(EventWrapper&& w) {
			func = std::move(w.func);
			return *this;
		}
		template<class... CallArgs>
		void operator()(CallArgs&& ...args) {
			trim_call([this, args...](auto&&... args) {(*func)(std::forward<decltype(args)>(args)...); },
				TypeQueue<Args...>{}, std::forward<CallArgs>(args)...);
		}
		Func_Type& Type() const {
			return func->ft;
		}
		bool operator==(const EventWrapper& ew) const{
			if (this->Type() != ew.Type()) return false;
			Func_Type tp = this->Type();
			
			std::vector<Base_Function<Args...>::addr_type> addr_vl = func->get_address();
			std::vector<Base_Function<Args...>::addr_type> addr_vr = ew.func->get_address();
			switch (tp) {
			case Func_Type::Lambda: { return false; }break;//���������lambda���󣬾�Ϊ��ͬ�¼�
			case Func_Type::MenberFunction: {
			if (addr_vl[1] == addr_vr[1] && addr_vl[2] == addr_vr[2]) return true;
			   else return false;
			}break;
			case Func_Type::GlobalFunction: {
				if (addr_vl[1] == addr_vr[1]) return true;
				else return false;
			}break;
			}
			return false;
		}
		std::size_t hash() const {
			Func_Type tp = this->Type();
			std::vector<Base_Function<Args...>::addr_type> addr_v = func->get_address();
			switch (tp) {
			case Func_Type::Lambda: { 	
				std::size_t func_hash = std::hash<std::size_t>()(addr_v[1]);
				std::size_t f_hash = std::hash<std::size_t>()(addr_v[0]);
				return ((func_hash % factor )* (f_hash % factor)) % factor;
			}break;
			case Func_Type::MenberFunction: {
				std::size_t func_hash = std::hash<std::size_t>()(addr_v[1]);
				std::size_t obj_hash = std::hash<std::size_t>()(addr_v[2]);
				 return ((func_hash % factor)* (obj_hash % factor))% factor;
			}break;
			case Func_Type::GlobalFunction: {
				std::size_t func_hash = std::hash<std::size_t>()(addr_v[1]);
				return func_hash % factor;
			}break;
			}
			return addr_v[0] % factor;
		}
	private:
		constexpr static std::size_t factor = 2654435769;
		std::unique_ptr<Base_Function<Args...>> func;
	};

	template<class... Args>
	class Base_Signal {
	public:
		using iterator =typename std::list<std::shared_ptr< EventWrapper<Args...>>>::iterator;
		Base_Signal() = default;
		template<class... F>
		void connect(F&&... f) {
			std::shared_ptr<EventWrapper<Args...>> e_now=std::make_shared<EventWrapper<Args...>>(std::forward<F>(f)...);
			if (event_map.find(e_now) == event_map.end()) {
				mes_q.push_back(e_now);
				iterator it = mes_q.end();
				event_map.insert({e_now ,--it });
			}
		}
		template<class... F>
		void disconnect(F&&... f){
			std::shared_ptr<EventWrapper<Args...>> e_now = std::make_shared<EventWrapper<Args...>>(std::forward<F>(f)...);
			if (event_map.find(e_now) != event_map.end()) {
				iterator it = event_map[e_now];
				mes_q.erase(it);
				event_map.erase(e_now);			
			}
		}
		template<class... CallArgs>
		void emit(CallArgs&&... args) {
			for (auto it = mes_q.begin(); it != mes_q.end();it++) {
				(*(*it))(std::forward<CallArgs>(args)...);
			}
		}
		struct Eventequal {
			bool operator()(const  std::shared_ptr<EventWrapper<Args...>> & ew_l, const std::shared_ptr < EventWrapper<Args...>>& ew_r) const {
				return  (*ew_l).operator==(*ew_r);
			}
		};
		struct EventHash {
			std::size_t operator()(const std::shared_ptr<EventWrapper<Args...>>& ew) const {
				return ew->hash();
			}
		};
	private:
		std::list<std::shared_ptr<EventWrapper<Args...>>> mes_q;
		std::unordered_map< std::shared_ptr<EventWrapper<Args...>>, iterator, EventHash, Eventequal> event_map;
	};
}
#endif
