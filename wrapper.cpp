// wrapper.cpp

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "LimitOrderBook.h"

namespace py = pybind11;

PYBIND11_MODULE(fast_lob, m)
{
    m.doc() = "High-performance C++ Limit Order Book for Python";
    py::class_<Trade>(m, "Trade")
        .def_readonly("buyer_id", &Trade::buyer_id)
        .def_readonly("seller_id", &Trade::seller_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity);
    py::class_<LimitOrderBook>(m, "LimitOrderBook")
        .def(py::init<>())

        .def("add_limit_order", &LimitOrderBook::add_limit_order,
             py::arg("is_buy_side"), py::arg("price"), py::arg("agent_id"), py::arg("quantity"))

        .def("process_market_order", &LimitOrderBook::process_market_order,
             py::arg("is_buy_side"), py::arg("quantity"), py::arg("agent_id"))

        .def("cancel_order", &LimitOrderBook::cancel_order)

        .def("get_best_bid", &LimitOrderBook::get_best_bid)
        .def("get_best_ask", &LimitOrderBook::get_best_ask)
        .def("clear_agent_orders", &LimitOrderBook::clear_agent_orders, py::arg("agent_id"), py::arg("fee_per_order"));
}