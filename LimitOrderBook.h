// LimitOrderBook.h

#pragma once

#include <cstdint>
#include <string>
#include <map>
#include <list>
#include <vector>
#include <unordered_map>
#include <functional>
#include <algorithm>
using Price = uint64_t;
struct Order;
using OrderQueue = std::list<Order>;
using OrderIterator = OrderQueue::iterator;
struct Order
{
    uint64_t order_id;
    uint32_t quantity;
    std::string agent_id;
    Price price;
    bool is_buy_side;
};
struct Trade
{
    std::string buyer_id;
    std::string seller_id;
    Price price;
    uint32_t quantity;
};
class LimitOrderBook
{
public:
    std::map<Price, OrderQueue> asks;
    std::map<Price, OrderQueue, std::greater<Price>> bids;
    std::unordered_map<uint64_t, OrderIterator> order_map;
    double clear_agent_orders(std::string agent_id, double fee_per_order);
    LimitOrderBook() {}

    void add_limit_order(bool is_buy_side, Price price, std::string agent_id, uint32_t quantity);
    std::vector<Trade> process_market_order(bool is_buy_side, uint32_t quantity, std::string agent_id);
    void cancel_order(uint64_t order_id);
    Price get_best_bid();
    Price get_best_ask();

private:
    uint64_t next_order_id = 0;
};