#define BOOST_SIMD_NO_STRICT_ALIASING 1
#include <pythonic/core.hpp>
#include <pythonic/python/core.hpp>
#include <pythonic/types/bool.hpp>
#include <pythonic/types/int.hpp>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <pythonic/include/types/numpy_texpr.hpp>
#include <pythonic/include/types/complex128.hpp>
#include <pythonic/include/types/ndarray.hpp>
#include <pythonic/include/types/float64.hpp>
#include <pythonic/include/types/float.hpp>
#include <pythonic/types/ndarray.hpp>
#include <pythonic/types/complex128.hpp>
#include <pythonic/types/float.hpp>
#include <pythonic/types/float64.hpp>
#include <pythonic/types/numpy_texpr.hpp>
#include <pythonic/include/__builtin__/dict.hpp>
#include <pythonic/include/types/slice.hpp>
#include <pythonic/include/operator_/div.hpp>
#include <pythonic/include/operator_/idiv.hpp>
#include <pythonic/include/__builtin__/list.hpp>
#include <pythonic/include/__builtin__/None.hpp>
#include <pythonic/include/types/str.hpp>
#include <pythonic/__builtin__/dict.hpp>
#include <pythonic/types/slice.hpp>
#include <pythonic/operator_/div.hpp>
#include <pythonic/operator_/idiv.hpp>
#include <pythonic/__builtin__/list.hpp>
#include <pythonic/__builtin__/None.hpp>
#include <pythonic/types/str.hpp>
namespace __pythran__pseudo_spect
{
  struct arguments_blocks
  {
    typedef void callable;
    typedef void pure;
    struct type
    {
      typedef pythonic::types::str __type0;
      typedef pythonic::types::list<typename std::remove_reference<__type0>::type> __type1;
      typedef typename pythonic::returnable<pythonic::types::dict<__type0,__type1>>::type result_type;
    }  ;
    typename type::result_type operator()() const;
    ;
  }  ;
  struct rk4_step3
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type1;
      typedef long __type2;
      typedef decltype((pythonic::operator_::div(std::declval<__type1>(), std::declval<__type2>()))) __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
      typedef decltype((std::declval<__type3>() * std::declval<__type4>())) __type5;
      typedef decltype((std::declval<__type0>() + std::declval<__type5>())) __type6;
      typedef __type6 __ptype0;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& tendencies_3, argument_type3&& dt) const
    ;
  }  ;
  struct rk4_step2
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type0;
      typedef long __type1;
      typedef decltype((pythonic::operator_::div(std::declval<__type0>(), std::declval<__type1>()))) __type2;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type5>::type>::type __type3;
      typedef decltype((std::declval<__type2>() * std::declval<__type3>())) __type4;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type5;
      typedef decltype((std::declval<__type4>() * std::declval<__type5>())) __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type7;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type8;
      typedef decltype((std::declval<__type7>() * std::declval<__type8>())) __type9;
      typedef decltype((std::declval<__type0>() * std::declval<__type3>())) __type10;
      typedef decltype((std::declval<__type10>() * std::declval<__type5>())) __type11;
      typedef decltype((std::declval<__type9>() + std::declval<__type11>())) __type12;
      typedef __type6 __ptype1;
      typedef __type12 __ptype2;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6>::result_type operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& state_spect_np1_approx, argument_type3&& tendencies_2, argument_type4&& diss, argument_type5&& diss2, argument_type6&& dt) const
    ;
  }  ;
  struct rk4_step1
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type5>::type>::type __type0;
      typedef long __type1;
      typedef decltype((pythonic::operator_::div(std::declval<__type0>(), std::declval<__type1>()))) __type2;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type3;
      typedef decltype((std::declval<__type2>() * std::declval<__type3>())) __type4;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type5;
      typedef decltype((std::declval<__type4>() * std::declval<__type5>())) __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type7;
      typedef decltype((std::declval<__type7>() * std::declval<__type3>())) __type8;
      typedef decltype((std::declval<__type2>() * std::declval<__type5>())) __type9;
      typedef decltype((std::declval<__type8>() + std::declval<__type9>())) __type10;
      typedef __type6 __ptype3;
      typedef __type10 __ptype4;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5>::result_type operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& state_spect_np12_approx2, argument_type3&& tendencies_1, argument_type4&& diss2, argument_type5&& dt) const
    ;
  }  ;
  struct rk4_step0
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type6>::type>::type __type1;
      typedef long __type2;
      typedef decltype((pythonic::operator_::div(std::declval<__type1>(), std::declval<__type2>()))) __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
      typedef decltype((std::declval<__type3>() * std::declval<__type4>())) __type5;
      typedef decltype((std::declval<__type0>() + std::declval<__type5>())) __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type7;
      typedef decltype((std::declval<__type6>() * std::declval<__type7>())) __type8;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type5>::type>::type __type9;
      typedef decltype((std::declval<__type6>() * std::declval<__type9>())) __type10;
      typedef __type8 __ptype5;
      typedef __type10 __ptype6;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6>::result_type operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& tendencies_0, argument_type3&& state_spect_np12_approx1, argument_type4&& diss, argument_type5&& diss2, argument_type6&& dt) const
    ;
  }  ;
  struct rk2_step1
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type0>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type1;
      typedef decltype((std::declval<__type0>() * std::declval<__type1>())) __type2;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type4;
      typedef decltype((std::declval<__type3>() * std::declval<__type4>())) __type5;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type6;
      typedef decltype((std::declval<__type5>() * std::declval<__type6>())) __type7;
      typedef decltype((std::declval<__type2>() + std::declval<__type7>())) __type8;
      typedef __type8 __ptype7;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type operator()(argument_type0&& state_spect, argument_type1&& tendencies_n12, argument_type2&& diss, argument_type3&& diss2, argument_type4&& dt) const
    ;
  }  ;
  struct rk2_step0
  {
    typedef void callable;
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    struct type
    {
      typedef typename std::remove_cv<typename std::remove_reference<argument_type1>::type>::type __type0;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type4>::type>::type __type1;
      typedef long __type2;
      typedef decltype((pythonic::operator_::div(std::declval<__type1>(), std::declval<__type2>()))) __type3;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type2>::type>::type __type4;
      typedef decltype((std::declval<__type3>() * std::declval<__type4>())) __type5;
      typedef decltype((std::declval<__type0>() + std::declval<__type5>())) __type6;
      typedef typename std::remove_cv<typename std::remove_reference<argument_type3>::type>::type __type7;
      typedef decltype((std::declval<__type6>() * std::declval<__type7>())) __type8;
      typedef __type8 __ptype8;
      typedef typename pythonic::returnable<pythonic::types::none_type>::type result_type;
    }  
    ;
    template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
    typename type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type operator()(argument_type0&& state_spect_n12, argument_type1&& state_spect, argument_type2&& tendencies_n, argument_type3&& diss2, argument_type4&& dt) const
    ;
  }  ;
  typename arguments_blocks::type::result_type arguments_blocks::operator()() const
  {
    {
      static typename arguments_blocks::type::result_type tmp_global = typename pythonic::assignable<pythonic::types::dict<pythonic::types::str,pythonic::types::list<typename std::remove_reference<pythonic::types::str>::type>>>::type{{{ "rk2_step0", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("tendencies_n")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_n12")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type>::type>::type>>::type({"state_spect_n12", "state_spect", "tendencies_n", "diss2", "dt"}) }, { "rk2_step1", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("tendencies_n12")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type>::type>::type>>::type({"state_spect", "tendencies_n12", "diss", "diss2", "dt"}) }, { "rk4_step0", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_np12_approx1")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_tmp")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("tendencies_0")>::type>::type>::type>::type>>::type({"state_spect", "state_spect_tmp", "tendencies_0", "state_spect_np12_approx1", "diss", "diss2", "dt"}) }, { "rk4_step1", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("state_spect_np12_approx2")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_tmp")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("tendencies_1")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type>::type>::type>>::type({"state_spect", "state_spect_tmp", "state_spect_np12_approx2", "tendencies_1", "diss2", "dt"}) }, { "rk4_step2", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("tendencies_2")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_np1_approx")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_tmp")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("diss")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type>::type>::type>>::type({"state_spect", "state_spect_tmp", "state_spect_np1_approx", "tendencies_2", "diss", "diss2", "dt"}) }, { "rk4_step3", typename pythonic::assignable<pythonic::types::list<typename std::remove_reference<typename __combined<typename __combined<typename __combined<typename std::remove_cv<typename std::remove_reference<decltype("state_spect")>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("state_spect_tmp")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("dt")>::type>::type>::type,typename std::remove_cv<typename std::remove_reference<decltype("tendencies_3")>::type>::type>::type>::type>>::type({"state_spect", "state_spect_tmp", "tendencies_3", "dt"}) }}};
      return tmp_global;
    }
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 >
  typename rk4_step3::type<argument_type0, argument_type1, argument_type2, argument_type3>::result_type rk4_step3::operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& tendencies_3, argument_type3&& dt) const
  {
    state_spect[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = (state_spect_tmp + ((pythonic::operator_::div(dt, 6L)) * tendencies_3));
    return pythonic::__builtin__::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
  typename rk4_step2::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6>::result_type rk4_step2::operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& state_spect_np1_approx, argument_type3&& tendencies_2, argument_type4&& diss, argument_type5&& diss2, argument_type6&& dt) const
  {
    state_spect_tmp[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] += (((pythonic::operator_::div(dt, 3L)) * diss2) * tendencies_2);
    state_spect_np1_approx[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect * diss) + ((dt * diss2) * tendencies_2));
    return pythonic::__builtin__::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 >
  typename rk4_step1::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5>::result_type rk4_step1::operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& state_spect_np12_approx2, argument_type3&& tendencies_1, argument_type4&& diss2, argument_type5&& dt) const
  {
    state_spect_tmp[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] += (((pythonic::operator_::div(dt, 3L)) * diss2) * tendencies_1);
    state_spect_np12_approx2[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect * diss2) + ((pythonic::operator_::div(dt, 2L)) * tendencies_1));
    return pythonic::__builtin__::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 , typename argument_type5 , typename argument_type6 >
  typename rk4_step0::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4, argument_type5, argument_type6>::result_type rk4_step0::operator()(argument_type0&& state_spect, argument_type1&& state_spect_tmp, argument_type2&& tendencies_0, argument_type3&& state_spect_np12_approx1, argument_type4&& diss, argument_type5&& diss2, argument_type6&& dt) const
  {
    state_spect_tmp[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect + ((pythonic::operator_::div(dt, 6L)) * tendencies_0)) * diss);
    state_spect_np12_approx1[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect + ((pythonic::operator_::div(dt, 2L)) * tendencies_0)) * diss2);
    return pythonic::__builtin__::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
  typename rk2_step1::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type rk2_step1::operator()(argument_type0&& state_spect, argument_type1&& tendencies_n12, argument_type2&& diss, argument_type3&& diss2, argument_type4&& dt) const
  {
    state_spect[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect * diss) + ((dt * diss2) * tendencies_n12));
    return pythonic::__builtin__::None;
  }
  template <typename argument_type0 , typename argument_type1 , typename argument_type2 , typename argument_type3 , typename argument_type4 >
  typename rk2_step0::type<argument_type0, argument_type1, argument_type2, argument_type3, argument_type4>::result_type rk2_step0::operator()(argument_type0&& state_spect_n12, argument_type1&& state_spect, argument_type2&& tendencies_n, argument_type3&& diss2, argument_type4&& dt) const
  {
    state_spect_n12[pythonic::types::contiguous_slice(pythonic::__builtin__::None,pythonic::__builtin__::None)] = ((state_spect + ((pythonic::operator_::div(dt, 2L)) * tendencies_n)) * diss2);
    return pythonic::__builtin__::None;
  }
}
#include <pythonic/python/exception_handler.hpp>
#ifdef ENABLE_PYTHON_MODULE
static PyObject* arguments_blocks = to_python(__pythran__pseudo_spect::arguments_blocks()());
typename __pythran__pseudo_spect::rk4_step3::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, double>::result_type rk4_step30(pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_3, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step3()(state_spect, state_spect_tmp, tendencies_3, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step3::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk4_step31(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_3, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step3()(state_spect, state_spect_tmp, tendencies_3, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step20(pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_2, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step21(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk4_step22(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::ndarray<std::complex<double>,3>&& diss, pythonic::types::ndarray<std::complex<double>,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::ndarray<double,2>, double>::result_type rk4_step23(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::ndarray<double,2>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk4_step24(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::ndarray<double,2>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::ndarray<double,2>, double>::result_type rk4_step25(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step2::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk4_step26(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np1_approx, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_2, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step2()(state_spect, state_spect_tmp, state_spect_np1_approx, tendencies_2, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step1::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step10(pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_np12_approx2, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_1, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step1()(state_spect, state_spect_tmp, state_spect_np12_approx2, tendencies_1, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step11(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx2, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_1, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step1()(state_spect, state_spect_tmp, state_spect_np12_approx2, tendencies_1, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk4_step12(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx2, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_1, pythonic::types::ndarray<std::complex<double>,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step1()(state_spect, state_spect_tmp, state_spect_np12_approx2, tendencies_1, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, double>::result_type rk4_step13(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx2, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_1, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step1()(state_spect, state_spect_tmp, state_spect_np12_approx2, tendencies_1, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk4_step14(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx2, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_1, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step1()(state_spect, state_spect_tmp, state_spect_np12_approx2, tendencies_1, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step00(pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,4>&& state_spect_np12_approx1, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk4_step01(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk4_step02(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::ndarray<std::complex<double>,3>&& diss, pythonic::types::ndarray<std::complex<double>,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::ndarray<double,2>, double>::result_type rk4_step03(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::ndarray<double,2>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk4_step04(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::ndarray<double,2>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::ndarray<double,2>, double>::result_type rk4_step05(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk4_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk4_step06(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_tmp, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_0, pythonic::types::ndarray<std::complex<double>,3>&& state_spect_np12_approx1, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk4_step0()(state_spect, state_spect_tmp, tendencies_0, state_spect_np12_approx1, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk2_step10(pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_n12, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,3>, pythonic::types::ndarray<double,3>, double>::result_type rk2_step11(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::ndarray<double,3>&& diss, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk2_step12(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::ndarray<std::complex<double>,3>&& diss, pythonic::types::ndarray<std::complex<double>,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::ndarray<double,2>, double>::result_type rk2_step13(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::ndarray<double,2>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk2_step14(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::ndarray<double,2>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::ndarray<double,2>, double>::result_type rk2_step15(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step1::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk2_step16(pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n12, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step1()(state_spect, tendencies_n12, diss, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step0::type<pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<std::complex<double>,4>, pythonic::types::ndarray<double,3>, double>::result_type rk2_step00(pythonic::types::ndarray<std::complex<double>,4>&& state_spect_n12, pythonic::types::ndarray<std::complex<double>,4>&& state_spect, pythonic::types::ndarray<std::complex<double>,4>&& tendencies_n, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step0()(state_spect_n12, state_spect, tendencies_n, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,3>, double>::result_type rk2_step01(pythonic::types::ndarray<std::complex<double>,3>&& state_spect_n12, pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n, pythonic::types::ndarray<double,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step0()(state_spect_n12, state_spect, tendencies_n, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, double>::result_type rk2_step02(pythonic::types::ndarray<std::complex<double>,3>&& state_spect_n12, pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n, pythonic::types::ndarray<std::complex<double>,3>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step0()(state_spect_n12, state_spect, tendencies_n, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<double,2>, double>::result_type rk2_step03(pythonic::types::ndarray<std::complex<double>,3>&& state_spect_n12, pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n, pythonic::types::ndarray<double,2>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step0()(state_spect_n12, state_spect, tendencies_n, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}
typename __pythran__pseudo_spect::rk2_step0::type<pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::ndarray<std::complex<double>,3>, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>, double>::result_type rk2_step04(pythonic::types::ndarray<std::complex<double>,3>&& state_spect_n12, pythonic::types::ndarray<std::complex<double>,3>&& state_spect, pythonic::types::ndarray<std::complex<double>,3>&& tendencies_n, pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>&& diss2, double&& dt) 
{
  
                            PyThreadState *_save = PyEval_SaveThread();
                            try {
                                auto res = __pythran__pseudo_spect::rk2_step0()(state_spect_n12, state_spect, tendencies_n, diss2, dt);
                                PyEval_RestoreThread(_save);
                                return res;
                            }
                            catch(...) {
                                PyEval_RestoreThread(_save);
                                throw;
                            }
                            ;
}

static PyObject *
__pythran_wrap_rk4_step30(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[4+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_3","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]) && is_convertible<double>(args_obj[3]))
        return to_python(rk4_step30(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]), from_python<double>(args_obj[3])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step31(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[4+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_3","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<double>(args_obj[3]))
        return to_python(rk4_step31(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<double>(args_obj[3])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step20(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step20(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<pythonic::types::ndarray<double,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step21(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step21(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<pythonic::types::ndarray<double,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step22(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step22(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step23(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step23(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,2>>(args_obj[4]), from_python<pythonic::types::ndarray<double,2>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step24(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[4]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step24(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,2>>(args_obj[4]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step25(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step25(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]), from_python<pythonic::types::ndarray<double,2>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step26(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np1_approx","tendencies_2","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step26(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step10(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np12_approx2","tendencies_1","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<double>(args_obj[5]))
        return to_python(rk4_step10(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<double>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step11(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np12_approx2","tendencies_1","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<double>(args_obj[5]))
        return to_python(rk4_step11(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<double>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step12(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np12_approx2","tendencies_1","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]) && is_convertible<double>(args_obj[5]))
        return to_python(rk4_step12(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]), from_python<double>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step13(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np12_approx2","tendencies_1","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[4]) && is_convertible<double>(args_obj[5]))
        return to_python(rk4_step13(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,2>>(args_obj[4]), from_python<double>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step14(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[6+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","state_spect_np12_approx2","tendencies_1","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]) && is_convertible<double>(args_obj[5]))
        return to_python(rk4_step14(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]), from_python<double>(args_obj[5])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step00(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step00(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<pythonic::types::ndarray<double,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step01(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step01(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,3>>(args_obj[4]), from_python<pythonic::types::ndarray<double,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step02(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step02(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[4]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step03(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step03(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,2>>(args_obj[4]), from_python<pythonic::types::ndarray<double,2>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step04(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[4]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step04(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::ndarray<double,2>>(args_obj[4]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step05(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step05(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]), from_python<pythonic::types::ndarray<double,2>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk4_step06(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[7+1];
    char const* keywords[] = {"state_spect","state_spect_tmp","tendencies_0","state_spect_np12_approx1","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4], &args_obj[5], &args_obj[6]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]) && is_convertible<double>(args_obj[6]))
        return to_python(rk4_step06(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[4]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[5]), from_python<double>(args_obj[6])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step10(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step10(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<double,3>>(args_obj[2]), from_python<pythonic::types::ndarray<double,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step11(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step11(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<double,3>>(args_obj[2]), from_python<pythonic::types::ndarray<double,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step12(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step12(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step13(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step13(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<double,2>>(args_obj[2]), from_python<pythonic::types::ndarray<double,2>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step14(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[2]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step14(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<double,2>>(args_obj[2]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step15(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step15(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[2]), from_python<pythonic::types::ndarray<double,2>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step16(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect","tendencies_n12","diss","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[2]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step16(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[2]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step00(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect_n12","state_spect","tendencies_n","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step00(from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,4>>(args_obj[2]), from_python<pythonic::types::ndarray<double,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step01(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect_n12","state_spect","tendencies_n","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step01(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<double,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step02(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect_n12","state_spect","tendencies_n","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step02(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step03(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect_n12","state_spect","tendencies_n","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::ndarray<double,2>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step03(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::ndarray<double,2>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

static PyObject *
__pythran_wrap_rk2_step04(PyObject *self, PyObject *args, PyObject *kw)
{
    PyObject* args_obj[5+1];
    char const* keywords[] = {"state_spect_n12","state_spect","tendencies_n","diss2","dt", nullptr};
    if(! PyArg_ParseTupleAndKeywords(args, kw, "OOOOO",
                                     (char**)keywords, &args_obj[0], &args_obj[1], &args_obj[2], &args_obj[3], &args_obj[4]))
        return nullptr;
    if(is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]) && is_convertible<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]) && is_convertible<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]) && is_convertible<double>(args_obj[4]))
        return to_python(rk2_step04(from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[0]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[1]), from_python<pythonic::types::ndarray<std::complex<double>,3>>(args_obj[2]), from_python<pythonic::types::numpy_texpr<pythonic::types::ndarray<double,2>>>(args_obj[3]), from_python<double>(args_obj[4])));
    else {
        return nullptr;
    }
}

            static PyObject *
            __pythran_wrapall_rk4_step3(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk4_step30(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step31(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk4_step3", "   rk4_step3(complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],float)\n   rk4_step3(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_rk4_step2(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk4_step20(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step21(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step22(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step23(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step24(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step25(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step26(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk4_step2", "   rk4_step2(complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:],float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:].T,float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:],float)\n   rk4_step2(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:].T,float)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_rk4_step1(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk4_step10(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step11(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step12(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step13(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step14(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk4_step1", "   rk4_step1(complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],float64[:,:,:],float)\n   rk4_step1(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:,:],float)\n   rk4_step1(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)\n   rk4_step1(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float)\n   rk4_step1(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_rk4_step0(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk4_step00(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step01(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step02(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step03(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step04(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step05(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk4_step06(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk4_step0", "   rk4_step0(complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:],float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:].T,float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:],float)\n   rk4_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:].T,float)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_rk2_step1(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk2_step10(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step11(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step12(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step13(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step14(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step15(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step16(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk2_step1", "   rk2_step1(complex128[:,:,:,:],complex128[:,:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],float64[:,:,:],float64[:,:,:],float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:],float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],float64[:,:],float64[:,:].T,float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:],float)\n   rk2_step1(complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float64[:,:].T,float)", args, kw);
                });
            }


            static PyObject *
            __pythran_wrapall_rk2_step0(PyObject *self, PyObject *args, PyObject *kw)
            {
                return pythonic::handle_python_exception([self, args, kw]()
                -> PyObject* {

if(PyObject* obj = __pythran_wrap_rk2_step00(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step01(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step02(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step03(self, args, kw))
    return obj;
PyErr_Clear();


if(PyObject* obj = __pythran_wrap_rk2_step04(self, args, kw))
    return obj;
PyErr_Clear();

                return pythonic::python::raise_invalid_argument(
                               "rk2_step0", "   rk2_step0(complex128[:,:,:,:],complex128[:,:,:,:],complex128[:,:,:,:],float64[:,:,:],float)\n   rk2_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:,:],float)\n   rk2_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float)\n   rk2_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:],float)\n   rk2_step0(complex128[:,:,:],complex128[:,:,:],complex128[:,:,:],float64[:,:].T,float)", args, kw);
                });
            }


static PyMethodDef Methods[] = {
    {
    "rk4_step3",
    (PyCFunction)__pythran_wrapall_rk4_step3,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk4_step3(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], float)\n    - rk4_step3(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)"},{
    "rk4_step2",
    (PyCFunction)__pythran_wrapall_rk4_step2,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk4_step2(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:], float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:].T, float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:], float)\n    - rk4_step2(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:].T, float)"},{
    "rk4_step1",
    (PyCFunction)__pythran_wrapall_rk4_step1,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk4_step1(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], float64[:,:,:], float)\n    - rk4_step1(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:,:], float)\n    - rk4_step1(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)\n    - rk4_step1(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float)\n    - rk4_step1(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float)"},{
    "rk4_step0",
    (PyCFunction)__pythran_wrapall_rk4_step0,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk4_step0(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:], float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:].T, float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:], float)\n    - rk4_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:].T, float)"},{
    "rk2_step1",
    (PyCFunction)__pythran_wrapall_rk2_step1,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk2_step1(complex128[:,:,:,:], complex128[:,:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], float64[:,:,:], float64[:,:,:], float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:], float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], float64[:,:], float64[:,:].T, float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:], float)\n    - rk2_step1(complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float64[:,:].T, float)"},{
    "rk2_step0",
    (PyCFunction)__pythran_wrapall_rk2_step0,
    METH_VARARGS | METH_KEYWORDS,
    "Supported prototypes:\n\n    - rk2_step0(complex128[:,:,:,:], complex128[:,:,:,:], complex128[:,:,:,:], float64[:,:,:], float)\n    - rk2_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:,:], float)\n    - rk2_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float)\n    - rk2_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:], float)\n    - rk2_step0(complex128[:,:,:], complex128[:,:,:], complex128[:,:,:], float64[:,:].T, float)"},
    {NULL, NULL, 0, NULL}
};


#if PY_MAJOR_VERSION >= 3
  static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "_pseudo_spect",            /* m_name */
    "",         /* m_doc */
    -1,                  /* m_size */
    Methods,             /* m_methods */
    NULL,                /* m_reload */
    NULL,                /* m_traverse */
    NULL,                /* m_clear */
    NULL,                /* m_free */
  };
#define PYTHRAN_RETURN return theModule
#define PYTHRAN_MODULE_INIT(s) PyInit_##s
#else
#define PYTHRAN_RETURN return
#define PYTHRAN_MODULE_INIT(s) init##s
#endif
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_pseudo_spect)(void)
#ifndef _WIN32
__attribute__ ((visibility("default")))
__attribute__ ((externally_visible))
#endif
;
PyMODINIT_FUNC
PYTHRAN_MODULE_INIT(_pseudo_spect)(void) {
    #ifdef PYTHONIC_TYPES_NDARRAY_HPP
        import_array()
    #endif
    #if PY_MAJOR_VERSION >= 3
    PyObject* theModule = PyModule_Create(&moduledef);
    #else
    PyObject* theModule = Py_InitModule3("_pseudo_spect",
                                         Methods,
                                         ""
    );
    #endif
    if(! theModule)
        PYTHRAN_RETURN;
    PyObject * theDoc = Py_BuildValue("(sss)",
                                      "0.8.6",
                                      "2018-10-08 14:30:42.146613",
                                      "88344ec7963eff727224e7ca1d3a946b187068230d8700d5b4b0200bf970fbbe");
    if(! theDoc)
        PYTHRAN_RETURN;
    PyModule_AddObject(theModule,
                       "__pythran__",
                       theDoc);

    PyModule_AddObject(theModule, "arguments_blocks", arguments_blocks);
    PYTHRAN_RETURN;
}

#endif