
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
#include<queue>
namespace mysig {
/*
*  obj_function_traitsUsed to match class member function pointers,
lambda objects, but not free function pointers
Parameter type: R--return type C--functor type Args--formal 
parameter list type
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
	// singleton template class that can control its own lifetime 
	template<class T>
	class singleton {
	public:
		singleton(const singleton&) = delete;
		singleton( singleton&&) = delete;
		singleton& operator=(const singleton&) = delete;
		singleton& operator=(singleton&&) = delete;
		template<class... Args>
		static void Initialize(Args&&... args) {
			if (!instance) {
				std::lock_guard<std::mutex> lock(mut);
				  if (!instance) {
					  instance = std::make_shared<T>(std::forward<Args>(args)...);
				  }
			}
		}
		template<class... Args>
		static std::shared_ptr<T> GetInstance(Args&&... args) {
			Initialize(std::forward<Args>(args)...);
			return instance;
		}
		static std::shared_ptr<T> TryInstance() {
			return instance;
		}
		static void destroy() {
			std::lock_guard<std::mutex> lock(mut);
			instance.reset();
		}
	protected:
		static std::shared_ptr<T> instance;
		static std::mutex mut;
		singleton() = default;
	};
	template<class T>
	std::shared_ptr<T> singleton<T>::instance = nullptr;
	template<class T>
	std::mutex singleton<T>::mut;
// usage: is_oneofAll<T,Args...>,Determine whether T is one of the Args... type
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
	template<class Ty, class... Args>
	constexpr bool is_Callable = Callable< Ty, Args...>::value;
	/* TypeQueue you can perform the following operations on the types 
	in the list: Pop the head of the queue, pop the tail of the queue, 
	push the head of the queue, push the tail of the queue, and find the 
	element with the subscript k
	*/
	template<class... Ty>
	struct TypeQueue {};
	template<class Queue>
	struct EmptyType {};
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
	// void TypeQueue
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
	template<>
	struct EmptyType<TypeQueue<>> {
		static constexpr bool value = true;
	};

	template<class Queue>
	constexpr bool is_empty_v = EmptyType<Queue>::value;
	/*QueueCompertor<QueueLeft, QueueRight,Operation> for each element of QueuLeft 
	and QueuRight Check its type, if each element of QueuLeft is the same as QueuRight,
	then the value is 1, and the comparator will extract Type, if each element of 
	QueuLeft is the same as QueuRight, the type is is_empty<>, otherwise the first one is 
	different Element pairs of QueueType<L,R>
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
	struct convert_operation {//Determine whether the type U can be converted to T
		static constexpr bool value = std::is_convertible_v<U, T>;
	};
	template<class QueueLeft, class QueueRight, template<class T, class U>class Operation>
	constexpr bool QueueCompertor_v = QueueComperator<QueueLeft, QueueRight, Operation>::value;
	template<class QueueLeft, class QueueRight>
	constexpr bool is_QueueType_include = QueueCompertor_v< QueueLeft, QueueRight, convert_operation>;
	template<class QueueLeft, class QueueRight>
	constexpr bool is_QueueType_same = is_QueueType_include<QueueLeft, QueueRight> && is_QueueType_include<QueueRight, QueueLeft>;
	/*Extract the parameter list type of the callable object, the operator() of the callable 
	object F has the form of R(C::*)(Args...)The return type is TypeQueue<Args...>
	 */
	template<class T, class = void>
	struct function_Args {};
	template<class T>
	struct function_Args<T> {
		using type = typename function_Args<decltype(&T::operator())>::type;
	};
	template<class R, class T, class... Args>
	struct function_Args<R(T::*)(Args...) const> {//The operator() function of the lambda object is of const type
		using type = TypeQueue<Args...>;
	};
	template<class R, class... Args>
	struct function_Args<R(Args...)> {
		using type = TypeQueue<Args...>;
	};
	template<class T>
	using function_Args_t = typename function_Args<T>::type;
	template<class... CallArgs>
	struct member_funtion_trait{
		template<class... F>
		constexpr static auto check(F&&... f) ->std::void_t<> {}
		template<class R, class C, class... Args>
		constexpr static auto check(R(C::* f)(Args...), C* ptr) -> C {}
		constexpr static bool is_member_funtion_v(CallArgs&&... args) {
			return !std::is_same_v< std::void_t<>, decltype(check(std::forward<CallArgs>(args)...))>;
		}
		constexpr static auto getType(CallArgs&&... args)->decltype(check(std::forward<CallArgs>(args)...)) {}
	};	
	/*参数裁剪相关
	* 使用 trim_call()进行裁剪调用
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
		//Determine whether  CallArgs... types can be implicitly converted to Args...
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
		static_assert(ParamTrim_t::template value<Args...>, "CallArgs can't be converted to Args or has different length");
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
		using addr_type =  std::uintptr_t;
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
	struct  EventId
	{
		Func_Type tp;
		std::size_t hash_v;
		std::vector<std::uintptr_t> addr;
		std::function<bool(const EventId&)> op_equal;
		template<class... F>
		EventId(const Func_Type& t, std::size_t h,const std::vector<std::uintptr_t>& v,F&&... f)
			:tp(t),hash_v(h),addr(v), op_equal(std::forward<F>(f)...){}
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
		void operator()(CallArgs&& ...args) const{
			trim_call([this, args...](auto&&... args) {(*func)(std::forward<decltype(args)>(args)...); },
				TypeQueue<Args...>{}, std::forward<CallArgs>(args)...);
		}
		Func_Type& Type() const {
			return func->ft;
		}
		bool operator==(const EventWrapper& ew) const{
			std::vector<Base_Function<Args...>::addr_type> addr_vl = func->get_address();
			std::vector<Base_Function<Args...>::addr_type> addr_vr = ew.func->get_address();
			return isEqual(this->Type(), ew.Type(),addr_vl, addr_vr);
		}
		bool isEqual(const Func_Type& tp, const Func_Type& tp_r,const std::vector<typename Base_Function<Args...>::addr_type>&  addr_vl,const std::vector<typename Base_Function<Args...>::addr_type>&  addr_vr) const{
			if (tp != tp_r) return false;
			switch (tp) {
			case Func_Type::Lambda: { return false; }break;//Any lambda object is a different event
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
		EventId GetEventId() {
			return EventId(this->Type(), this->hash(), func->get_address(), [&](const EventId& e)
				{return isEqual(this->Type(), e.tp, func->get_address(), e.addr); });
		}
	private:
		constexpr static std::size_t factor = 2654435769;
		std::unique_ptr<Base_Function<Args...>> func;
	};
	template<class... Args>
	struct Eventequal {
		bool operator()(const  std::shared_ptr<EventWrapper<Args...>>& ew_l, const std::shared_ptr < EventWrapper<Args...>>& ew_r) const {
			return  (*ew_l).operator==(*ew_r);
		}
	};
	template<class... Args>
	struct EventHash {
		std::size_t operator()(const std::shared_ptr<EventWrapper<Args...>>& ew) const {
			return ew->hash();
		}
	};
	//A partial specialization for EventId
	template<>
	struct Eventequal<EventId> {
		bool operator()(const EventId& ew_l, const EventId& ew_r) const {
			return  ew_l.op_equal(ew_r);
		}
	};
	template<>
	struct EventHash <EventId> {
		std::size_t operator()(const EventId& ew) const {
			return ew.hash_v;
		}
	};
	template<class... Args>
	class Slot;
	class Object;
	template<class... Args>
	class Base_Signal {
	public:
		friend class Slot<Args...>;
		using iterator =typename std::list<std::shared_ptr< EventWrapper<Args...>>>::iterator;
		Base_Signal() = default;
		~Base_Signal() {
			std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::TryInstance();
			if (!slot) return;
			for (auto&& p : event_map) {
				slot->disconnect(p.first, this);
			}
			if (!slot->has_signal()) {
				Slot<Args...>::destroy();
			}
		}
		template<class... F>
		std::shared_ptr<EventWrapper<Args...>> connect(F&&... f) {
			std::shared_ptr<EventWrapper<Args...>> e_now=std::make_shared<EventWrapper<Args...>>(std::forward<F>(f)...);
			if (event_map.find(e_now) == event_map.end()) {
				mes_q.push_back(e_now);
				iterator it = mes_q.end();
				event_map.insert({e_now ,--it });
			}
			return e_now;
		}
		template<class... F>
		std::shared_ptr<EventWrapper<Args...>> disconnect(F&&... f){
			std::shared_ptr<EventWrapper<Args...>> e_now = std::make_shared<EventWrapper<Args...>>(std::forward<F>(f)...);
			remove(e_now);
			return e_now;
		}
		template<class... CallArgs>
		void emit(CallArgs&&... args) {
			for (auto it = mes_q.begin(); it != mes_q.end();it++) {
				(*(*it))(std::forward<CallArgs>(args)...);
			}
		}
	private:
		bool remove(const std::shared_ptr<EventWrapper<Args...>>& ew) {
			if (event_map.find(ew) != event_map.end()) {
				iterator it = event_map[ew];
				mes_q.erase(it);
				event_map.erase(ew);
				return true;
			}
			else return false;
		}
		std::list<std::shared_ptr<EventWrapper<Args...>>> mes_q;
		std::unordered_map< std::shared_ptr<EventWrapper<Args...>>, iterator, EventHash<Args...>, Eventequal<Args...>> event_map;
	};
	template<class... Args>
	class Slot : public singleton<Slot<Args...>> {
	public:
		Slot() = default;
		~Slot() {
			std::cout << "SLot destroy" << std::endl;
		}
		void connect(const std::shared_ptr<EventWrapper<Args...>>& ew, Base_Signal<Args...>* sig) {
			std::unordered_set<Base_Signal<Args...>*>& sig_set = sig_con[ew];
			sig_set.insert(sig);
		}
		void disconnect(const std::shared_ptr<EventWrapper<Args...>>& ew, Base_Signal<Args...>* sig) {
			std::unordered_set<Base_Signal<Args...>*>& sig_set = sig_con[ew];
			if (sig_set.find(sig) == sig_set.end()) {
				throw std::invalid_argument("can't disconnect signal doesn't exit");
			}
			sig_set.erase(sig);
			if (sig_set.empty()) {
				sig_con.erase(ew);
			}
		}
		void disconnect_all(const std::shared_ptr<EventWrapper<Args...>>& ew) {
			std::unordered_set<Base_Signal<Args...>*>& sig_set = sig_con[ew];
			for (Base_Signal<Args...>* it : sig_set) {
				it->remove(ew);
			}
			sig_con.erase(ew);
		}
		bool has_signal() const { return !sig_con.empty(); }
	private:
		std::unordered_map<std::shared_ptr<EventWrapper<Args...>>, std::unordered_set<Base_Signal<Args...>*>, EventHash<Args...>, Eventequal<Args...>> sig_con;
	};
	template<class... Args,class... Left>
	void connect(Base_Signal<Args...>* sig, Left&&... left) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::GetInstance();
		slot->connect(sig->connect(std::forward<Left>(left)...),sig);
	}
	template<class R,class C,class... Args>
	void connect(Base_Signal<Args...>* sig,R(C::*func)(Args...),C* obj) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::GetInstance();
		std::shared_ptr<EventWrapper<Args...>> ew = sig->connect(func,obj);
		if constexpr (std::is_base_of_v<Object, C>) {
			static_cast<Object*>(obj)->registe(ew);
		}
		slot->connect(ew, sig);
	}
	template< class... Args, class... Left>
	void disconnect( Base_Signal<Args...>* sig, Left&&... left) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::TryInstance();
		if (!slot) return;
		slot->disconnect(sig->disconnect(std::forward<Left>(left)...),sig);
		if (!slot->has_signal()){
			Slot<Args...>::destroy(); 
		}
	}
	template<class R, class C, class... Args>
	void disconnect(Base_Signal<Args...>* sig, R(C::* func)(Args...), C* obj) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::TryInstance();
		if (!slot) return;
		std::shared_ptr<EventWrapper<Args...>> ew = sig->connect(func,obj);
		if constexpr (std::is_base_of_v<Object, C>) {
			static_cast<Object*>(obj)->cancle(ew);
		}
		slot->disconnect(ew, sig);
		if (!slot->has_signal()) {
			Slot<Args...>::destroy();
		}
	}
	template<class C,class R,class... Args>
	void disconnect_all(R(C::* func_ptr)(Args...), C* obj_ptr) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::TryInstance();
		if (!slot) return;
		std::shared_ptr<EventWrapper<Args...>> e_now = std::make_shared<EventWrapper<Args...>>(func_ptr,obj_ptr);
		slot->disconnect_all(e_now);
		if (!slot->has_signal()) {
			Slot<Args...>::destroy();
		}
	}
	template< class... Args>
	void disconnect_all(const std::shared_ptr<EventWrapper<Args...>>& ev) {
		std::shared_ptr<Slot<Args...>> slot = Slot<Args...>::TryInstance();
		if (!slot) return;
		slot->disconnect_all(ev);
		if (!slot->has_signal()) {
			Slot<Args...>::destroy();
		}
	}
	enum class LoopPolicy {
		Continue,
		Asynic_user,
		Threadpool,
	};
// one thread per loop. The creation of EventLoop is completed by external thread 
// because  thread holds all of resource of  EventLoop. EventLoop must be 
// designed to allow to its own thread to carry on task and other threads
// to steal task from the loop.
//
	class EventLoop {
	public: 
		explicit EventLoop(LoopPolicy lp):is_run(false), lp(lp), thread_id(std::this_thread::get_id()) {}
		EventLoop():is_run(false), lp(LoopPolicy::Continue), thread_id(std::this_thread::get_id()){}
		~EventLoop() {
			if (lp == LoopPolicy::Threadpool) {
				std::unique_lock<std::mutex> lock(mut);
				is_run = false;
				cv.notify_all();
			}
		}
		void loop() {
			while (is_run) {
				std::unique_lock<std::mutex> lock(mut);
				cv.wait(lock, [&]() {return !is_run || !event_list.empty(); });
				EventWrapper<> callback = std::move( event_list.front());
				event_list.pop();
				lock.unlock();
				callback();
			}
		} 
		template<class... Args,class... CallArgs>
		void submit(Base_Signal<Args...>& sig, CallArgs&&... args) {
			if (lp == LoopPolicy::Continue) {
				for (auto& ew : sig.event_map) {
					(ew.first)(std::forward<CallArgs>(args)...);
				}
			}
			else if (lp == LoopPolicy::Threadpool) {
				for (auto& ew : sig.event_map) {
					this->put(ew, std::forward<CallArgs>(args)...);
				}
			}
		}
		template<class... Args,class... CallArgs>
		void put(const std::shared_ptr<EventWrapper<Args...>>& ew, CallArgs&&... args) {
			if (std::this_thread::get_id() == thread_id) {
				event_list.emplace([ew, args...]() {(*ew)(std::forward<CallArgs>(args)...); });
				cv.notify_one();
			}
			else {
				std::unique_lock<std::mutex> lock(mut);
			}
			
			
		}

	private:
		std::thread::id thread_id;
		std::queue<EventWrapper<>> event_list;
		std::mutex mut;
		std::condition_variable cv;
		bool is_run;
		LoopPolicy lp;
	};
	class EventThread {
	public:
		EventThread(std::function<void(EventLoop*)> c):callback(c), 
			th(&EventThread::excute,this){}
		std::shared_ptr<EventLoop> getEventLoop() {
			std::shared_ptr<EventLoop> t_loop = loop.lock();
			if (!t_loop) {
				std::unique_lock<std::mutex> lock(mut);
				cv.wait(lock, [&]() {t_loop=loop.lock(); return t_loop!=nullptr; });
			}
			return t_loop;
		}
	private:

		void excute() {
			std::shared_ptr<EventLoop> now_loop(new EventLoop());
			if (callback) {
				if (auto loop_ptr = loop.lock()) {
					callback(loop_ptr.get());
				}
			}
			std::unique_lock<std::mutex> lock(mut);
			loop = now_loop;
			cv.notify_one();
			now_loop->loop();
		}
		std::thread th;
		std::mutex mut;
		std::condition_variable cv;
		std::weak_ptr<EventLoop> loop;
		std::function<void(EventLoop*)> callback;
 	};
	class EventLoopPool {
	public:
		EventLoopPool(int c):capacity(c){}
		void start() {
			for (int i = 0; i < capacity; ++i) {
				std::unique_ptr<EventThread> t(new EventThread());
			}
		}
		void registe(EventLoop* loop) {
			std::thread::id nowid = std::this_thread::get_id();
			if (loop_map.find(nowid) != loop_map.end()) {
				loop_map.insert({ nowid ,loop });
			}
		}
	private:
		int capacity;
		std::vector<std::unique_ptr<EventThread>> threads;
		std::unordered_map<std::thread::id,EventLoop*> loop_map;
	};
/*Automatically manage the managed base class of the connection, when the target class
* inherits the Object class, the target is destroyed When the destructor of Object is 
* called, all signals and slots are disconnected Note that registe and cancle do not 
* directly handle connection and disconnection logic, they should be handled by connect
* and diconnect direct call
* */
	class Object {
	public:
		virtual ~Object(){
			for (auto&& ew_pair : ew_map) {
				(ew_pair.second)();
			}
		}
		template<class... Args>
		void registe(const std::shared_ptr < EventWrapper<Args...>>& ew) {	
			EventId id = ew->GetEventId();
			if (ew_map.find(id) == ew_map.end()) {
				ew_map.insert({ std::move(id), EventWrapper<>([=]() {disconnect_all(ew); }) });
			}	
		}
		template<class... Args>
		void cancle(const std::shared_ptr < EventWrapper<Args...>>& ew) {
			EventId id = ew->GetEventId();
			if (ew_map.find(id) != ew_map.end()) {
				ew_map.erase(id);
			}
		}

		std::unordered_map<EventId, EventWrapper<>, EventHash<EventId>, Eventequal<EventId>> ew_map;
	};
}
#endif
