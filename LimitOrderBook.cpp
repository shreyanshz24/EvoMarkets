// LimitOrderBook.cpp
#include "LimitOrderBook.h"
#include <vector>
#include <algorithm>

void LimitOrderBook::add_limit_order(bool is_buy_side, Price price, std::string agent_id, uint32_t quantity)
{

    uint64_t new_order_id = next_order_id++;

    Order new_order;
    new_order.order_id = new_order_id;
    new_order.agent_id = agent_id;
    new_order.quantity = quantity;
    new_order.price = price;
    new_order.is_buy_side = is_buy_side;

    if (is_buy_side)
    {
        OrderQueue &queue = bids[price];
        queue.push_back(new_order);
        OrderIterator it = --queue.end();
        order_map[new_order_id] = it;
    }
    else
    {
        OrderQueue &queue = asks[price];
        queue.push_back(new_order);
        OrderIterator it = --queue.end();
        order_map[new_order_id] = it;
    }
}
std::vector<Trade> LimitOrderBook::process_market_order(bool is_buy_side, uint32_t quantity, std::string agent_id)
{

    std::vector<Trade> trades_made;
    uint32_t quantity_needed = quantity;

    if (is_buy_side)
    {
        while (quantity_needed > 0 && !asks.empty())
        {

            auto best_level_it = asks.begin();
            Price level_price = best_level_it->first;
            OrderQueue &queue = best_level_it->second;

            auto order_it = queue.begin();
            while (order_it != queue.end() && quantity_needed > 0)
            {

                Order &best_order = *order_it;
                if (best_order.agent_id == agent_id)
                {
                    order_it++;
                    continue;
                }

                uint32_t trade_quantity = std::min(quantity_needed, best_order.quantity);

                trades_made.push_back(Trade{
                    agent_id,
                    best_order.agent_id,
                    level_price,
                    trade_quantity});

                quantity_needed -= trade_quantity;
                best_order.quantity -= trade_quantity;

                if (best_order.quantity == 0)
                {
                    order_map.erase(best_order.order_id);
                    order_it = queue.erase(order_it);
                }
                else
                {
                    order_it++;
                }
            }
            if (queue.empty())
            {
                asks.erase(best_level_it);
            }
        }
    }
    else
    {
        while (quantity_needed > 0 && !bids.empty())
        {

            auto best_level_it = bids.begin();
            Price level_price = best_level_it->first;
            OrderQueue &queue = best_level_it->second;

            auto order_it = queue.begin();
            while (order_it != queue.end() && quantity_needed > 0)
            {

                Order &best_order = *order_it;
                if (best_order.agent_id == agent_id)
                {
                    order_it++;
                    continue;
                }

                uint32_t trade_quantity = std::min(quantity_needed, best_order.quantity);

                trades_made.push_back(Trade{
                    best_order.agent_id,
                    agent_id,
                    level_price,
                    trade_quantity});

                quantity_needed -= trade_quantity;
                best_order.quantity -= trade_quantity;

                if (best_order.quantity == 0)
                {
                    order_map.erase(best_order.order_id);
                    order_it = queue.erase(order_it);
                }
                else
                {
                    order_it++;
                }
            }
            if (queue.empty())
            {
                bids.erase(best_level_it);
            }
        }
    }
    return trades_made;
}
void LimitOrderBook::cancel_order(uint64_t order_id)
{
    // 1. Find the order in the lookup map
    auto it = order_map.find(order_id);
    if (it == order_map.end())
    {
        return;
    }
    OrderIterator order_it = it->second;
    Price price = order_it->price;
    bool is_buy = order_it->is_buy_side;
    if (is_buy)
    {
        auto level_it = bids.find(price);
        if (level_it != bids.end())
        {
            OrderQueue &queue = level_it->second;
            queue.erase(order_it);
            if (queue.empty())
            {
                bids.erase(level_it);
            }
        }
    }
    else
    {
        auto level_it = asks.find(price);
        if (level_it != asks.end())
        {
            OrderQueue &queue = level_it->second;
            queue.erase(order_it);
            if (queue.empty())
            {
                asks.erase(level_it);
            }
        }
    }
    order_map.erase(it);
}
double LimitOrderBook::clear_agent_orders(std::string agent_id, double fee_per_order)
{
    double total_fee = 0.0;

    std::vector<uint64_t> ids_to_cancel;

    // 1. Find all orders belonging to this agent
    for (auto const &[order_id, order_iterator] : order_map)
    {
        if (order_iterator->agent_id == agent_id)
        {
            ids_to_cancel.push_back(order_id);
        }
    }
    for (uint64_t id : ids_to_cancel)
    {
        this->cancel_order(id);
        total_fee += fee_per_order;
    }

    return total_fee;
}
Price LimitOrderBook::get_best_bid()
{
    return bids.empty() ? 0 : bids.begin()->first;
}
Price LimitOrderBook::get_best_ask()
{
    return asks.empty() ? 1000000000 : asks.begin()->first;
}