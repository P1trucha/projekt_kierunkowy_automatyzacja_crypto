def size_by_balance_usdt(balance_usdt, risk_frac=0.02, price=0.0):
    # 2% kapitału w ryzyku to nie to samo co 2% pozycji, ale na start uprośćmy
    if price <= 0:
        return 0.0
    notional = balance_usdt * risk_frac
    qty = round(notional / price, 6)
    return max(qty, 0.0)
